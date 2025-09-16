// static/js/dashboard.js (模擬與 UI 強化版)

document.addEventListener('DOMContentLoaded', () => {
    // --- WebSocket 連線 ---
    const socket = io();

    // --- 狀態管理 ---
    let modelLoaded = false;
    let activeView = 'dashboard-view';
    const MAX_CHART_POINTS = 200; // 圖表保留的最大點數

    let simulationUpdateQueue = [];
    let isRenderLoopRunning = false;
    let lastUpdateClientTS = null;
    let updateFrequencyEMA = null;
    let latencyEMA = null;

    // --- DOM 元素 ---
    const views = document.querySelectorAll('.view');
    const navButtons = document.querySelectorAll('.nav-button');
    const sidebarContent = document.getElementById('sidebar-content');
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

    // =================================================================================
    // 視圖切換
    // =================================================================================

    function showView(viewId) {
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
        sidebarContent.innerHTML = activeView === 'training-view' ? trainingSidebarTemplate : dashboardSidebarTemplate;
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
                plugins: { legend: { labels: { color: '#d1d5db' } } },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }

    function resetSimulationView() {
        simulationUpdateQueue = [];
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
            updateChartData(simulationChart, labels, [predictedData, actualData, errorData, voltageData]);
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

    function updateChartData(chart, labels, datasetsData) {
        if (!chart) return;
        for (let i = 0; i < labels.length; i++) {
            chart.data.labels.push(labels[i]);
            chart.data.datasets.forEach((dataset, idx) => {
                dataset.data.push(datasetsData[idx][i]);
            });
            if (chart.data.labels.length > MAX_CHART_POINTS) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
        }
        chart.update('quiet');
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
        if (!trainingChart) return;
        const progress = (data.epoch / data.total_epochs) * 100;
        const progressBar = document.getElementById('progress-bar');
        const epochCounter = document.getElementById('epoch-counter');
        if (progressBar) progressBar.style.width = `${progress}%`;
        if (epochCounter) {
            epochCounter.innerHTML = `<p class="text-gray-300 text-sm">Epoch</p><p class="text-white font-bold text-lg">${data.epoch} / ${data.total_epochs}</p>`;
        }
        updateChartData(trainingChart, [data.epoch], [[data.train_loss], [data.val_loss], [data.val_rmse]]);
    });

    socket.on('training_finished', (data) => {
        modelLoaded = data.model_loaded;
        if (ui.retrainBtn) {
            ui.retrainBtn.disabled = false;
            ui.retrainBtn.textContent = '重新訓練模型';
        }
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
        startRenderLoop();
    });

    socket.on('simulation_update', (data) => {
        if (data && data.points && data.points.length > 0) {
            simulationUpdateQueue.push(...data.points);
            updatePerformanceIndicators(data);
        }
    });

    socket.on('simulation_stopped', () => {
        stopRenderLoop();
        updateSimulationAvailability();
    });

    socket.on('simulation_error', (data) => {
        updateStatus(data.message || '模擬發生錯誤。', 'error');
        stopRenderLoop();
        updateSimulationAvailability();
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

    navButtons.forEach(btn => btn.addEventListener('click', () => showView(btn.dataset.view)));
    showView('dashboard-view');
    initializeTrainingChart();
    initializeSimulationChart();
});
