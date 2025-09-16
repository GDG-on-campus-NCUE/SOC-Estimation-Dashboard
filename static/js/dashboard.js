// static/js/dashboard.js (最終性能優化版)

document.addEventListener('DOMContentLoaded', () => {
    // --- WebSocket 連接 ---
    const socket = io();

    // --- 狀態管理 ---
    let modelLoaded = false;
    let activeView = 'dashboard-view'; // 預設視圖
    const MAX_CHART_POINTS = 100; // 圖表上顯示的最大數據點數量
    
    // --- 用於平滑渲染的客戶端緩衝區 ---
    let simulationUpdateQueue = [];
    let isRenderLoopRunning = false;

    // --- DOM 元素引用 ---
    const views = document.querySelectorAll('.view');
    const navButtons = document.querySelectorAll('.nav-button');
    const sidebarContent = document.getElementById('sidebar-content');
    const statusBar = {
        light: document.getElementById('status-light'),
        label: document.getElementById('status-label')
    };
    
    // --- 側邊欄模板 ---
    const trainingSidebarTemplate = document.getElementById('training-sidebar-template').innerHTML;
    const dashboardSidebarTemplate = document.getElementById('dashboard-sidebar-template').innerHTML;

    // --- 圖表實例 ---
    let trainingChart, simulationChart;

    // --- UI 元素 (在模板注入後會被填充) ---
    let ui = {};

    // =================================================================================
    // 初始化與視圖管理
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
        sidebarContent.innerHTML = activeView === 'training-view' ? trainingSidebarTemplate : dashboardSidebarTemplate;
        if (activeView === 'training-view') {
            bindTrainingControls();
        } else {
            bindDashboardControls();
            ui.startSimBtn.disabled = !modelLoaded;
            ui.stopSimBtn.disabled = true;
        }
    }

    // =================================================================================
    // UI 綁定與事件監聽
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

        // --- FIX: 樂觀更新 (Optimistic Update) ---
        ui.startSimBtn.addEventListener('click', () => {
            ui.startSimBtn.disabled = true;
            ui.stopSimBtn.disabled = false;
            socket.emit('start_simulation');
            startRenderLoop();
        });
        ui.stopSimBtn.addEventListener('click', () => {
            ui.startSimBtn.disabled = !modelLoaded;
            ui.stopSimBtn.disabled = true;
            socket.emit('stop_simulation');
            stopRenderLoop();
        });
        
        const paramInputs = [ui.speedSlider, ui.vNoise, ui.cNoise, ui.tNoise];
        paramInputs.forEach(input => input.addEventListener('input', updateSimulationParams));
    }

    function startTraining() {
        ui.retrainBtn.disabled = true;
        ui.retrainBtn.textContent = 'Training in Progress...';
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
        ui.speedLabel.textContent = `${parseFloat(ui.speedSlider.value).toFixed(1)}x`;
        socket.emit('update_simulation_params', {
            speed: ui.speedSlider.value, v_noise: ui.vNoise.value,
            c_noise: ui.cNoise.value, t_noise: ui.tNoise.value,
        });
    }

    // =================================================================================
    // 圖表繪製
    // =================================================================================

    function initializeTrainingChart() {
        if (trainingChart) trainingChart.destroy();
        const ctx = document.getElementById('training-chart').getContext('2d');
        trainingChart = new Chart(ctx, {
            type: 'line',
            data: { 
                labels: [], 
                datasets: [
                    { label: 'Training Loss', data: [], borderColor: 'rgba(56, 189, 248, 1)', yAxisID: 'yLoss', borderWidth: 2 },
                    { label: 'Validation Loss', data: [], borderColor: 'rgba(34, 197, 94, 1)', yAxisID: 'yLoss', borderWidth: 2 },
                    { label: 'Validation RMSE (%)', data: [], borderColor: 'rgba(234, 179, 8, 1)', yAxisID: 'yRMSE', borderWidth: 2, borderDash: [5, 5] }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Epoch', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' }},
                    yLoss: { type: 'logarithmic', position: 'left', title: { display: true, text: 'Loss (Log Scale)', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' }},
                    yRMSE: { type: 'linear', position: 'right', title: { display: true, text: 'RMSE (%)', color: '#9ca3af' }, ticks: { color: 'rgba(234, 179, 8, 1)' }, grid: { drawOnChartArea: false }}
                },
                plugins: {
                    legend: { labels: { color: '#d1d5db' } },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) { label += ': '; }
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
        if (simulationChart) simulationChart.destroy();
        const ctx = document.getElementById('simulation-chart').getContext('2d');
        simulationChart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [
                { label: 'Estimated SOC (%)', data: [], borderColor: '#22d3ee', backgroundColor: 'rgba(34, 211, 238, 0.1)', fill: true, yAxisID: 'ySOC', tension: 0.3 },
                { label: 'Voltage (V)', data: [], borderColor: '#60a5fa', yAxisID: 'yV', tension: 0.3 }
            ]},
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { title: { display: true, text: 'Time Step', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(156, 163, 175, 0.2)' }},
                    ySOC: { type: 'linear', position: 'left', min: 0, max: 100, title: { display: true, text: 'SOC (%)', color: '#9ca3af' }, ticks: { color: '#22d3ee' }, grid: { color: 'rgba(156, 163, 175, 0.2)' }},
                    yV: { type: 'linear', position: 'right', title: { display: true, text: 'Voltage (V)', color: '#9ca3af' }, ticks: { color: '#60a5fa' }, grid: { drawOnChartArea: false }}
                },
                plugins: { legend: { labels: { color: '#d1d5db' } } },
                interaction: { mode: 'index', intersect: false }
            }
        });
    }
    
    // =================================================================================
    // 高性能渲染迴圈
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
            const socData = pointsToRender.map(p => p.soc);
            const voltageData = pointsToRender.map(p => p.v);
            updateChartData(simulationChart, labels, [socData, voltageData]);
        }

        requestAnimationFrame(renderLoop);
    }

    function updateDashboardMetrics(data) {
        document.getElementById('soc-value').textContent = data.soc.toFixed(2);
        document.getElementById('v-value').textContent = data.v.toFixed(2);
        document.getElementById('c-value').textContent = data.c.toFixed(2);
        document.getElementById('t-value').textContent = data.t.toFixed(1);

        const socCircle = document.getElementById('soc-progress-circle');
        const radius = socCircle.r.baseVal.value;
        const circumference = 2 * Math.PI * radius;
        socCircle.style.strokeDashoffset = circumference - (data.soc / 100) * circumference;
    }

    function updateChartData(chart, labels, datasetsData) {
        for (let i = 0; i < labels.length; i++) {
            chart.data.labels.push(labels[i]);
            for (let j = 0; j < datasetsData.length; j++) {
                chart.data.datasets[j].data.push(datasetsData[j][i]);
            }
            if (chart.data.labels.length > MAX_CHART_POINTS) {
                chart.data.labels.shift();
                chart.data.datasets.forEach(dataset => dataset.data.shift());
            }
        }
        chart.update('quiet');
    }
    
    // =================================================================================
    // SOCKET.IO 事件處理
    // =================================================================================

    socket.on('initial_state', (data) => {
        modelLoaded = data.model_loaded;
        updateSidebar();
        updateStatus(modelLoaded ? 'Pre-trained model loaded. Ready.' : 'No model. Please train first.', 'info');
    });

    socket.on('status', (data) => updateStatus(data.message, data.type));

    socket.on('training_update', (data) => {
        const progress = (data.epoch / data.total_epochs) * 100;
        document.getElementById('progress-bar').style.width = `${progress}%`;
        document.getElementById('epoch-counter').innerHTML = `<p class="text-gray-400 text-sm">Epoch</p><p class="text-white font-bold text-lg">${data.epoch} / ${data.total_epochs}</p>`;
        updateChartData(trainingChart, [data.epoch], [[data.train_loss], [data.val_loss], [data.val_rmse]]);
    });
    
    socket.on('training_finished', (data) => {
        modelLoaded = data.model_loaded;
        ui.retrainBtn.disabled = false;
        ui.retrainBtn.textContent = 'Re-train Model';
        showView('dashboard-view');
    });

    socket.on('simulation_update', (data) => {
        if (data.points && data.points.length > 0) {
            simulationUpdateQueue.push(...data.points);
        }
    });

    function updateStatus(message, type = 'info') {
        statusBar.label.textContent = message;
        statusBar.light.className = 'w-3 h-3 rounded-full transition-colors duration-500 ';
        switch (type) {
            case 'training': statusBar.light.classList.add('bg-blue-400'); break;
            case 'simulating': statusBar.light.classList.add('bg-green-400'); break;
            case 'success': statusBar.light.classList.add('bg-green-500'); break;
            case 'error': statusBar.light.classList.add('bg-red-500'); break;
            case 'info': default: statusBar.light.classList.add('bg-yellow-400'); break;
        }
    }
    
    // --- 初始設定 ---
    navButtons.forEach(btn => btn.addEventListener('click', () => showView(btn.dataset.view)));
    showView('dashboard-view');
    initializeTrainingChart();
    initializeSimulationChart();
});
