// static/js/dashboard.js (模擬與 UI 強化版)

document.addEventListener('DOMContentLoaded', () => {
    // --- WebSocket 連線 ---
    const socket = io();

    // --- 狀態管理 ---
    let modelLoaded = false;
    let activeView = 'dashboard-view';
    const MAX_TRAINING_POINTS = 200; // 訓練圖表最多保留的點數
    const MAX_VISIBLE_SIMULATION_POINTS = 200; // 模擬圖表視窗內的參考顯示點數上限

    let simulationUpdateQueue = [];
    let simulationHistory = [];
    let trainingHistory = [];
    let isRenderLoopRunning = false;
    let lastUpdateClientTS = null;
    let updateFrequencyEMA = null;
    let latencyEMA = null;

    // --- DOM 元素 ---
    const views = document.querySelectorAll('.view');
    const navButtons = document.querySelectorAll('.nav-button');
    const sidebarContent = document.getElementById('sidebar-content');
    const simulationChartScroll = document.getElementById('simulation-chart-scroll');
    const simulationChartInner = document.getElementById('simulation-chart-inner');
    const simulationLegend = document.getElementById('simulation-legend');
    const trainingExportButton = document.getElementById('export-training-csv-btn');
    const statusBar = {
        light: document.getElementById('status-light'),
        label: document.getElementById('status-label')
    };

    // --- 模板 ---
    const trainingSidebarTemplate = document.getElementById('training-sidebar-template').innerHTML;
    const dashboardSidebarTemplate = document.getElementById('dashboard-sidebar-template').innerHTML;

    // --- 圖表實例 ---
    let trainingChart;
    let simulationChart;

    // --- UI 元件參考 ---
    let ui = {};

    // --- 提示資訊控制 ---
    let activeTooltip = null;
    const tooltipHideTimers = new WeakMap();
    const tooltipPanelsByContext = new Map();
    const TOOLTIP_HIDE_DELAY = 120; // 延遲隱藏的時間（毫秒），提供更順暢的體驗

    function initializeTooltips(scope = document, context = 'global') {
        const tooltipGroups = scope.querySelectorAll('.tooltip-group');
        tooltipGroups.forEach(group => {
            if (group.dataset.tooltipBound === 'true') return;
            const trigger = group.querySelector('.tooltip-icon');
            const panel = group.querySelector('.tooltip-panel');
            if (!trigger || !panel) return;

            registerTooltipPanel(panel, context);
            document.body.appendChild(panel);
            panel.dataset.tooltipPortal = 'true';

            const showHandler = () => {
                clearTooltipHideTimer(panel);
                showTooltip(trigger, panel);
            };
            const hideWithDelay = () => scheduleTooltipHide(panel);
            const hideImmediately = () => hideActiveTooltip(panel);

            group.addEventListener('mouseenter', showHandler);
            group.addEventListener('mouseleave', hideWithDelay);
            trigger.addEventListener('focus', showHandler);
            trigger.addEventListener('blur', hideImmediately);
            panel.addEventListener('mouseenter', () => clearTooltipHideTimer(panel));
            panel.addEventListener('mouseleave', hideWithDelay);

            group.dataset.tooltipBound = 'true';
        });
    }

    function registerTooltipPanel(panel, context) {
        const ctx = context || 'global';
        panel.dataset.tooltipContext = ctx;
        if (!tooltipPanelsByContext.has(ctx)) {
            tooltipPanelsByContext.set(ctx, new Set());
        }
        tooltipPanelsByContext.get(ctx).add(panel);
    }

    function cleanupTooltipsByContext(context) {
        if (!context || !tooltipPanelsByContext.has(context)) return;
        const panels = tooltipPanelsByContext.get(context);
        panels.forEach(panel => {
            hideActiveTooltip(panel);
            clearTooltipHideTimer(panel);
            panel.remove();
        });
        tooltipPanelsByContext.delete(context);
    }

    function scheduleTooltipHide(panel) {
        if (!panel) return;
        clearTooltipHideTimer(panel);
        const timer = setTimeout(() => hideActiveTooltip(panel), TOOLTIP_HIDE_DELAY);
        tooltipHideTimers.set(panel, timer);
    }

    function clearTooltipHideTimer(panel) {
        if (!panel) return;
        const timer = tooltipHideTimers.get(panel);
        if (timer) {
            clearTimeout(timer);
            tooltipHideTimers.delete(panel);
        }
    }

    function showTooltip(trigger, panel) {
        if (!trigger || !panel) return;
        if (activeTooltip && activeTooltip.panel !== panel) {
            hideActiveTooltip(activeTooltip.panel);
        }

        activeTooltip = { trigger, panel };
        panel.dataset.visible = 'true';
        positionTooltip(trigger, panel);
    }

    function hideActiveTooltip(panel) {
        const targetPanel = panel || (activeTooltip && activeTooltip.panel);
        if (!targetPanel) return;

        clearTooltipHideTimer(targetPanel);
        targetPanel.dataset.visible = 'false';
        targetPanel.removeAttribute('data-placement');

        if (activeTooltip && activeTooltip.panel === targetPanel) {
            activeTooltip = null;
        }
    }

    function positionTooltip(trigger, panel) {
        if (!trigger || !panel) return;

        panel.style.left = '0px';
        panel.style.top = '0px';

        const triggerRect = trigger.getBoundingClientRect();
        const tooltipWidth = panel.offsetWidth;
        const tooltipHeight = panel.offsetHeight;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const margin = 12;

        const spaceLeft = triggerRect.left - margin;
        const spaceRight = viewportWidth - triggerRect.right - margin;
        const spaceAbove = triggerRect.top - margin;
        const spaceBelow = viewportHeight - triggerRect.bottom - margin;

        const triggerVerticalCenter = triggerRect.top + triggerRect.height / 2;
        const triggerHorizontalCenter = triggerRect.left + triggerRect.width / 2;

        let placement = 'right';
        let left;
        let top;

        if (spaceRight >= tooltipWidth || spaceRight >= spaceLeft) {
            placement = 'right';
            left = triggerRect.right + margin;
            top = triggerVerticalCenter - tooltipHeight / 2;
        } else if (spaceLeft >= tooltipWidth) {
            placement = 'left';
            left = triggerRect.left - margin - tooltipWidth;
            top = triggerVerticalCenter - tooltipHeight / 2;
        } else if (spaceBelow >= tooltipHeight || spaceBelow >= spaceAbove) {
            placement = 'bottom';
            left = triggerHorizontalCenter - tooltipWidth / 2;
            top = triggerRect.bottom + margin;
        } else {
            placement = 'top';
            left = triggerHorizontalCenter - tooltipWidth / 2;
            top = triggerRect.top - margin - tooltipHeight;
        }

        left = Math.min(Math.max(margin, left), viewportWidth - margin - tooltipWidth);
        top = Math.min(Math.max(margin, top), viewportHeight - margin - tooltipHeight);

        panel.dataset.placement = placement;
        panel.style.left = `${left}px`;
        panel.style.top = `${top}px`;

        if (placement === 'left' || placement === 'right') {
            const arrowOffset = triggerVerticalCenter - top;
            const minOffset = Math.min(tooltipHeight / 2, 12);
            const maxOffset = tooltipHeight - minOffset;
            const clampedOffset = Math.min(Math.max(arrowOffset, minOffset), maxOffset);
            panel.style.setProperty('--tooltip-arrow-y', `${clampedOffset}px`);
            panel.style.removeProperty('--tooltip-arrow-x');
        } else {
            const arrowOffset = triggerHorizontalCenter - left;
            const minOffset = Math.min(tooltipWidth / 2, 12);
            const maxOffset = tooltipWidth - minOffset;
            const clampedOffset = Math.min(Math.max(arrowOffset, minOffset), maxOffset);
            panel.style.setProperty('--tooltip-arrow-x', `${clampedOffset}px`);
            panel.style.removeProperty('--tooltip-arrow-y');
        }
    }

    function refreshTooltipPosition() {
        if (!activeTooltip || activeTooltip.panel.dataset.visible !== 'true') return;
        positionTooltip(activeTooltip.trigger, activeTooltip.panel);
    }

    window.addEventListener('scroll', refreshTooltipPosition, { passive: true });
    window.addEventListener('resize', refreshTooltipPosition);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            hideActiveTooltip();
        }
    });

    const mainContainer = document.querySelector('main');
    if (mainContainer) {
        mainContainer.addEventListener('scroll', refreshTooltipPosition, { passive: true });
    }

    if (sidebarContent) {
        sidebarContent.addEventListener('scroll', refreshTooltipPosition, { passive: true });
    }

    // =================================================================================
    // 視圖切換
    // =================================================================================

    function showView(viewId) {
        hideActiveTooltip();
        views.forEach(view => view.classList.add('hidden'));
        document.getElementById(viewId)?.classList.remove('hidden');

        navButtons.forEach(btn => {
            const isActive = btn.dataset.view === viewId;
            btn.classList.toggle('bg-cyan-600', isActive);
            btn.classList.toggle('text-white', isActive);
            btn.classList.toggle('bg-gray-700', !isActive);
            btn.classList.toggle('hover:bg-gray-600', !isActive);
            btn.classList.toggle('text-gray-300', !isActive);
        });

        activeView = viewId;
        updateSidebar();
    }

    function updateSidebar() {
        ui = {};
        hideActiveTooltip();
        cleanupTooltipsByContext('sidebar');
        sidebarContent.innerHTML = activeView === 'training-view' ? trainingSidebarTemplate : dashboardSidebarTemplate;
        initializeTooltips(sidebarContent, 'sidebar');
        if (activeView === 'training-view') {
            bindTrainingControls();
        } else {
            bindDashboardControls();
            updateSimulationAvailability();
            updateDashboardMetrics(null);
            updatePerformanceIndicators(null, true);
        }
    }

    // =================================================================================
    // UI 綁定與事件
    // =================================================================================

    function bindTrainingControls() {
        ui.retrainBtn = document.getElementById('retrain-btn');
        ui.retrainBtn.addEventListener('click', startTraining);
        if (trainingExportButton) {
            ui.trainingExportBtn = trainingExportButton;
            toggleExportAvailability(trainingHistory.length > 0, trainingExportButton);
        }
    }

    function bindDashboardControls() {
        ui.startSimBtn = document.getElementById('start-sim-btn');
        ui.stopSimBtn = document.getElementById('stop-sim-btn');
        ui.speedSlider = document.getElementById('speed-slider');
        ui.speedLabel = document.getElementById('speed-label');
        ui.vNoise = document.getElementById('v-noise');
        ui.cNoise = document.getElementById('c-noise');
        ui.tNoise = document.getElementById('t-noise');
        ui.socValue = document.getElementById('soc-value');
        ui.socCircle = document.getElementById('soc-progress-circle');
        ui.vValue = document.getElementById('v-value');
        ui.cValue = document.getElementById('c-value');
        ui.tValue = document.getElementById('t-value');
        ui.actualSocValue = document.getElementById('actual-soc-value');
        ui.errorValue = document.getElementById('error-value');
        ui.updateFrequency = document.getElementById('update-frequency');
        ui.latencyValue = document.getElementById('latency-value');
        ui.windowSizeHint = document.getElementById('window-size-hint');
        ui.exportCsvBtn = document.getElementById('export-csv-btn');

        ui.startSimBtn.addEventListener('click', () => {
            if (!modelLoaded) return;
            ui.startSimBtn.disabled = true;
            ui.startSimBtn.textContent = '啟動中...';
            ui.stopSimBtn.disabled = false;
            resetSimulationView();
            socket.emit('start_simulation');
        });

        ui.stopSimBtn.addEventListener('click', () => {
            ui.stopSimBtn.disabled = true;
            socket.emit('stop_simulation');
        });

        const paramInputs = [ui.speedSlider, ui.vNoise, ui.cNoise, ui.tNoise];
        paramInputs.forEach(input => input?.addEventListener('input', updateSimulationParams));
        updateSimulationParams();

        ui.exportCsvBtn?.addEventListener('click', exportSimulationData);
        toggleExportAvailability(simulationHistory.length > 0);
    }

    function startTraining() {
        ui.retrainBtn.disabled = true;
        ui.retrainBtn.textContent = '訓練進行中...';
        initializeTrainingChart();
        const params = {
            windowSize: document.getElementById('windowSize').value,
            hiddenLayers: document.getElementById('hiddenLayers').value,
            learningRate: document.getElementById('learningRate').value,
            epochs: document.getElementById('epochs').value,
        };
        socket.emit('start_training', params);
    }

    function updateSimulationParams() {
        if (!ui.speedSlider) return;
        const speedValue = parseFloat(ui.speedSlider.value || '1');
        ui.speedLabel.textContent = `${speedValue.toFixed(1)}x`;
        socket.emit('update_simulation_params', {
            speed: speedValue,
            v_noise: ui.vNoise?.value || 0,
            c_noise: ui.cNoise?.value || 0,
            t_noise: ui.tNoise?.value || 0,
        });
    }

    function updateSimulationAvailability() {
        if (!ui.startSimBtn || !ui.stopSimBtn) return;
        ui.startSimBtn.disabled = !modelLoaded;
        ui.startSimBtn.textContent = modelLoaded ? '開始模擬' : '等待模型';
        ui.stopSimBtn.disabled = true;
    }

    // =================================================================================
    // 圖表初始化
    // =================================================================================

    function initializeTrainingChart() {
        if (!document.getElementById('training-chart')) return;
        if (trainingChart) trainingChart.destroy();
        const ctx = document.getElementById('training-chart').getContext('2d');
        trainingHistory = [];
        toggleExportAvailability(false, trainingExportButton);
        trainingChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: '訓練損失', data: [], borderColor: 'rgba(56, 189, 248, 1)', yAxisID: 'yLoss', borderWidth: 2 },
                    { label: '驗證損失', data: [], borderColor: 'rgba(34, 197, 94, 1)', yAxisID: 'yLoss', borderWidth: 2 },
                    { label: '驗證 RMSE (%)', data: [], borderColor: 'rgba(234, 179, 8, 1)', yAxisID: 'yRMSE', borderWidth: 2, borderDash: [5, 5] }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Epoch', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' } },
                    yLoss: { type: 'logarithmic', position: 'left', title: { display: true, text: 'Loss (Log Scale)', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' } },
                    yRMSE: { type: 'linear', position: 'right', title: { display: true, text: 'RMSE (%)', color: '#9ca3af' }, ticks: { color: 'rgba(234, 179, 8, 1)' }, grid: { drawOnChartArea: false } }
                },
                plugins: {
                    legend: { labels: { color: '#d1d5db' } },
                    tooltip: {
                        callbacks: {
                            label: (context) => {
                                let label = context.dataset.label || '';
                                if (label) label += ': ';
                                if (context.parsed.y !== null) {
                                    if (context.dataset.yAxisID === 'yLoss') {
                                        label += context.parsed.y.toExponential(4);
                                    } else {
                                        label += context.parsed.y.toFixed(4);
                                    }
                                }
                                return label;
                            }
                        }
                    }
                },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }

    function initializeSimulationChart() {
        if (!document.getElementById('simulation-chart')) return;
        if (simulationChart) simulationChart.destroy();
        const ctx = document.getElementById('simulation-chart').getContext('2d');
        simulationChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    { label: '預測 SOC (%)', data: [], borderColor: '#22d3ee', backgroundColor: 'rgba(34, 211, 238, 0.12)', fill: true, yAxisID: 'ySOC', tension: 0.25 },
                    { label: '實際 SOC (%)', data: [], borderColor: '#f59e0b', borderDash: [6, 4], yAxisID: 'ySOC', tension: 0.25 },
                    { label: '預測誤差 (%)', data: [], borderColor: '#f87171', yAxisID: 'yErr', tension: 0.2 },
                    { label: '電壓 (V)', data: [], borderColor: '#60a5fa', yAxisID: 'yV', tension: 0.2 }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: '時間步 (Index)', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' } },
                    ySOC: { type: 'linear', position: 'left', min: 0, max: 100, title: { display: true, text: 'SOC (%)', color: '#9ca3af' }, ticks: { color: '#22d3ee' }, grid: { color: 'rgba(156, 163, 175, 0.2)' } },
                    yErr: { type: 'linear', position: 'right', suggestedMin: -10, suggestedMax: 10, title: { display: true, text: '誤差 (%)', color: '#9ca3af' }, ticks: { color: '#f87171' }, grid: { drawOnChartArea: false } },
                    yV: { type: 'linear', position: 'right', offset: true, title: { display: true, text: 'Voltage (V)', color: '#9ca3af' }, ticks: { color: '#60a5fa' }, grid: { drawOnChartArea: false } }
                },
                plugins: {
                    legend: { display: false }
                },
                interaction: { mode: 'index', intersect: false }
            }
        });
        buildSimulationLegend();
        adjustSimulationChartViewport();
    }

    function buildSimulationLegend() {
        if (!simulationLegend) return;
        simulationLegend.innerHTML = '';
        if (!simulationChart) return;

        simulationChart.data.datasets.forEach((dataset, index) => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'legend-toggle-button';

            const colorDot = document.createElement('span');
            colorDot.className = 'legend-color-dot';
            colorDot.style.backgroundColor = dataset.borderColor;
            button.appendChild(colorDot);

            const label = document.createElement('span');
            label.textContent = dataset.label;
            button.appendChild(label);

            const setActiveState = (isActive) => {
                button.dataset.active = String(isActive);
                button.setAttribute('aria-pressed', String(isActive));
            };

            setActiveState(simulationChart.isDatasetVisible(index));

            button.addEventListener('click', () => {
                const currentlyVisible = simulationChart.isDatasetVisible(index);
                simulationChart.setDatasetVisibility(index, !currentlyVisible);
                setActiveState(!currentlyVisible);
                simulationChart.update('none');
            });

            simulationLegend.appendChild(button);
        });
    }

    function resetSimulationView() {
        simulationUpdateQueue = [];
        simulationHistory = [];
        updateDashboardMetrics(null);
        updatePerformanceIndicators(null, true);
        if (ui.windowSizeHint) {
            ui.windowSizeHint.textContent = '--';
        }
        if (simulationChart) {
            simulationChart.data.labels = [];
            simulationChart.data.datasets.forEach(dataset => { dataset.data = []; });
            simulationChart.update('none');
        }
        adjustSimulationChartViewport();
        toggleExportAvailability(false);
    }

    // =================================================================================
    // 渲染迴圈
    // =================================================================================

    function startRenderLoop() {
        if (isRenderLoopRunning) return;
        isRenderLoopRunning = true;
        requestAnimationFrame(renderLoop);
    }

    function stopRenderLoop() {
        isRenderLoopRunning = false;
    }

    function renderLoop() {
        if (!isRenderLoopRunning) return;

        const pointsToRender = simulationUpdateQueue.splice(0, simulationUpdateQueue.length);
        if (pointsToRender.length > 0) {
            const lastPoint = pointsToRender[pointsToRender.length - 1];
            updateDashboardMetrics(lastPoint);

            const labels = pointsToRender.map(p => p.index);
            const predictedData = pointsToRender.map(p => p.soc);
            const actualData = pointsToRender.map(p => p.actual_soc);
            const errorData = pointsToRender.map(p => p.error);
            const voltageData = pointsToRender.map(p => p.v);
            updateChartData(
                simulationChart,
                labels,
                [predictedData, actualData, errorData, voltageData],
                { keepAll: true }
            );
        }

        requestAnimationFrame(renderLoop);
    }

    function updateDashboardMetrics(data) {
        if (!ui.socValue) return;

        if (!data) {
            setText(ui.socValue, '--');
            setText(ui.vValue, '-.--');
            setText(ui.cValue, '-.--');
            setText(ui.tValue, '-.--');
            setText(ui.actualSocValue, '--');
            setText(ui.errorValue, '--');
            updateGauge(0);
            return;
        }

        const formattedSOC = formatNumber(data.soc, 2, '--');
        const formattedVoltage = formatNumber(data.v, 2, '-.--');
        const formattedCurrent = formatNumber(data.c, 2, '-.--');
        const formattedTemp = formatNumber(data.t, 1, '-.--');
        const formattedActual = formatNumber(data.actual_soc, 2, '--');
        const errorValue = typeof data.error === 'number' ? data.error : NaN;
        const formattedError = Number.isFinite(errorValue) ? `${errorValue >= 0 ? '+' : ''}${errorValue.toFixed(2)}%` : '--';

        setText(ui.socValue, formattedSOC);
        setText(ui.vValue, formattedVoltage);
        setText(ui.cValue, formattedCurrent);
        setText(ui.tValue, formattedTemp);
        setText(ui.actualSocValue, formattedActual);
        setText(ui.errorValue, formattedError);
        updateGauge(parseFloat(formattedSOC) || 0);
    }

    function setText(target, value) {
        if (target) target.textContent = value;
    }

    function formatNumber(value, decimals, fallback) {
        if (typeof value !== 'number' || Number.isNaN(value)) return fallback;
        return value.toFixed(decimals);
    }

    function updateGauge(socValue) {
        if (!ui.socCircle) return;
        const radius = ui.socCircle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        const safeSoc = Math.max(0, Math.min(100, socValue));
        ui.socCircle.style.strokeDashoffset = circumference - (safeSoc / 100) * circumference;
    }

    function updatePerformanceIndicators(meta, reset = false) {
        if (reset) {
            lastUpdateClientTS = null;
            updateFrequencyEMA = null;
            latencyEMA = null;
            setText(ui.updateFrequency, '--');
            setText(ui.latencyValue, '--');
            return;
        }

        const now = performance.now();
        if (lastUpdateClientTS) {
            const deltaMs = now - lastUpdateClientTS;
            if (deltaMs > 0) {
                const freq = 1000 / deltaMs;
                updateFrequencyEMA = updateFrequencyEMA === null ? freq : updateFrequencyEMA * 0.7 + freq * 0.3;
            }
        }
        lastUpdateClientTS = now;

        if (meta && typeof meta.server_ts === 'number') {
            const latency = Math.max(0, now - meta.server_ts * 1000);
            latencyEMA = latencyEMA === null ? latency : latencyEMA * 0.7 + latency * 0.3;
        } else if (meta && typeof meta.latency_hint_ms === 'number') {
            const latency = Math.max(0, meta.latency_hint_ms);
            latencyEMA = latencyEMA === null ? latency : latencyEMA * 0.7 + latency * 0.3;
        }

        if (ui.updateFrequency && updateFrequencyEMA !== null) {
            setText(ui.updateFrequency, `${updateFrequencyEMA.toFixed(1)} Hz`);
        }
        if (ui.latencyValue && latencyEMA !== null) {
            setText(ui.latencyValue, `${latencyEMA.toFixed(0)} ms`);
        }
    }

    function updateChartData(chart, labels, datasetsData, options = {}) {
        if (!chart) return;
        const { keepAll = false, maxPoints = MAX_TRAINING_POINTS } = options;
        for (let i = 0; i < labels.length; i++) {
            chart.data.labels.push(labels[i]);
            chart.data.datasets.forEach((dataset, idx) => {
                dataset.data.push(datasetsData[idx][i]);
            });
            if (!keepAll && chart.data.labels.length > maxPoints) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
        }
        chart.update('quiet');
        if (chart === simulationChart) {
            adjustSimulationChartViewport();
        }
    }

    function adjustSimulationChartViewport() {
        if (!simulationChartInner) return;

        const totalPoints = simulationChart?.data.labels.length || 0;
        const visibleLimit = Math.max(1, MAX_VISIBLE_SIMULATION_POINTS);
        const widthRatio = totalPoints > visibleLimit ? totalPoints / visibleLimit : 1;
        const previousScrollLeft = simulationChartScroll ? simulationChartScroll.scrollLeft : 0;
        const isNearRightEdge = simulationChartScroll
            ? (simulationChartScroll.scrollLeft + simulationChartScroll.clientWidth) >= (simulationChartScroll.scrollWidth - 4)
            : false;

        simulationChartInner.style.width = `${Math.max(1, widthRatio) * 100}%`;

        if (simulationChart) {
            simulationChart.resize();
        }

        if (simulationChartScroll) {
            const maxScrollLeft = Math.max(0, simulationChartScroll.scrollWidth - simulationChartScroll.clientWidth);
            if (isNearRightEdge) {
                simulationChartScroll.scrollLeft = maxScrollLeft;
            } else {
                simulationChartScroll.scrollLeft = Math.min(previousScrollLeft, maxScrollLeft);
            }
        }
    }

    function toggleExportAvailability(isEnabled, targetButton = null) {
        const button = targetButton || ui.exportCsvBtn;
        if (!button) return;
        button.disabled = !isEnabled;
        button.setAttribute('aria-disabled', String(!isEnabled));
    }

    function exportSimulationData() {
        if (!simulationHistory.length) return;

        const headers = ['index', 'predicted_soc', 'actual_soc', 'error', 'voltage', 'current', 'temperature'];
        const rows = simulationHistory.map(point => [
            point.index,
            formatCsvNumber(point.soc),
            formatCsvNumber(point.actual_soc),
            formatCsvNumber(point.error),
            formatCsvNumber(point.v),
            formatCsvNumber(point.c),
            formatCsvNumber(point.t)
        ].join(','));

        const csvContent = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const downloadLink = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
        downloadLink.href = url;
        downloadLink.download = `soc_simulation_${timestamp}.csv`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(url);
    }

    function exportTrainingData() {
        if (!trainingHistory.length) return;

        const headers = ['epoch', 'train_loss', 'val_loss', 'val_rmse'];
        const rows = trainingHistory.map(entry => [
            entry.epoch,
            formatTrainingCsvValue(entry.train_loss),
            formatTrainingCsvValue(entry.val_loss),
            formatTrainingCsvValue(entry.val_rmse)
        ].join(','));

        const csvContent = [headers.join(','), ...rows].join('\n');
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const downloadLink = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace('T', '_').split('.')[0];
        downloadLink.href = url;
        downloadLink.download = `training_metrics_${timestamp}.csv`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        URL.revokeObjectURL(url);
    }

    function formatCsvNumber(value, decimals = 4) {
        const num = Number(value);
        return Number.isFinite(num) ? num.toFixed(decimals) : '';
    }

    function formatTrainingCsvValue(value) {
        const num = Number(value);
        if (!Number.isFinite(num)) return '';
        const absVal = Math.abs(num);
        if (absVal === 0) return '0';
        if (absVal < 1e-4 || absVal >= 1e4) {
            return num.toExponential(6);
        }
        return num.toFixed(6);
    }

    // =================================================================================
    // SOCKET.IO 事件
    // =================================================================================

    socket.on('initial_state', (data) => {
        modelLoaded = data.model_loaded;
        updateSidebar();
        updateStatus(modelLoaded ? '已載入預訓練模型，可立即模擬。' : '尚未找到模型，請先完成訓練。', 'info');
    });

    socket.on('status', (data) => updateStatus(data.message, data.type));

    socket.on('training_update', (data) => {
        const latestEntry = trainingHistory[trainingHistory.length - 1];
        const newEntry = {
            epoch: data.epoch,
            train_loss: data.train_loss,
            val_loss: data.val_loss,
            val_rmse: data.val_rmse
        };

        if (latestEntry && latestEntry.epoch === data.epoch) {
            trainingHistory[trainingHistory.length - 1] = newEntry;
        } else {
            trainingHistory.push(newEntry);
        }

        toggleExportAvailability(trainingHistory.length > 0, trainingExportButton);

        const progress = (data.epoch / data.total_epochs) * 100;
        const progressBar = document.getElementById('progress-bar');
        const epochCounter = document.getElementById('epoch-counter');
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (epochCounter) {
            epochCounter.innerHTML = `<p class="text-gray-300 text-sm">Epoch</p><p class="text-white font-bold text-lg">${data.epoch} / ${data.total_epochs}</p>`;
        }

        if (trainingChart) {
            updateChartData(trainingChart, [data.epoch], [[data.train_loss], [data.val_loss], [data.val_rmse]]);
        }
    });

    socket.on('training_finished', (data) => {
        modelLoaded = data.model_loaded;
        if (ui.retrainBtn) {
            ui.retrainBtn.disabled = false;
            ui.retrainBtn.textContent = '重新訓練模型';
        }
        toggleExportAvailability(trainingHistory.length > 0, trainingExportButton);
        updateStatus('Training completed successfully.', 'success');
        showView('dashboard-view');
    });

    socket.on('simulation_started', (data) => {
        if (ui.windowSizeHint) {
            ui.windowSizeHint.textContent = data.window_size ?? '--';
        }
        if (ui.startSimBtn) {
            ui.startSimBtn.disabled = true;
            ui.startSimBtn.textContent = '模擬中...';
        }
        if (ui.stopSimBtn) {
            ui.stopSimBtn.disabled = false;
        }
        updatePerformanceIndicators(null, true);
        toggleExportAvailability(false);
        startRenderLoop();
    });

    socket.on('simulation_update', (data) => {
        if (data && data.points && data.points.length > 0) {
            simulationUpdateQueue.push(...data.points);
            simulationHistory.push(...data.points.map(point => ({ ...point })));
            updatePerformanceIndicators(data);
            toggleExportAvailability(simulationHistory.length > 0);
        }
    });

    socket.on('simulation_stopped', () => {
        stopRenderLoop();
        updateSimulationAvailability();
        toggleExportAvailability(simulationHistory.length > 0);
    });

    socket.on('simulation_error', (data) => {
        updateStatus(data.message || '模擬發生錯誤。', 'error');
        stopRenderLoop();
        updateSimulationAvailability();
        toggleExportAvailability(simulationHistory.length > 0);
    });

    // =================================================================================
    // 共同工具
    // =================================================================================

    function updateStatus(message, type = 'info') {
        statusBar.label.textContent = message;
        statusBar.light.className = 'w-3 h-3 rounded-full transition-colors duration-500 ';
        switch (type) {
            case 'training':
                statusBar.light.classList.add('bg-blue-400');
                break;
            case 'simulating':
                statusBar.light.classList.add('bg-green-400');
                break;
            case 'success':
                statusBar.light.classList.add('bg-green-500');
                break;
            case 'error':
                statusBar.light.classList.add('bg-red-500');
                break;
            case 'info':
            default:
                statusBar.light.classList.add('bg-yellow-400');
        }
    }

    // =================================================================================
    // 初始化
    // =================================================================================

    trainingExportButton?.addEventListener('click', exportTrainingData);

    navButtons.forEach(btn => btn.addEventListener('click', () => showView(btn.dataset.view)));
    showView('dashboard-view');
    initializeTrainingChart();
    initializeSimulationChart();
    initializeTooltips(document.body);
});
