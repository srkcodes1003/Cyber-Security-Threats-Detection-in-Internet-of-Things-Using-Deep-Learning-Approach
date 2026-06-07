/* ===================================================
   AETHER-SOC JavaScript Controller
   Chart.js updates, dynamic form layout, live simulation
   =================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // -------------------------------------------------
    // STATE VARIABLES
    // -------------------------------------------------
    let isStreaming = false;
    let streamIntervalId = null;
    let streamIntervalTime = 500; // ms
    let chartInstance = null;
    let chartDataBuffer = {
        labels: [],
        confidence: [],
        latency: []
    };
    
    // Telemetry aggregators
    let totalPackets = 0;
    let totalThreats = 0;
    let sumLatency = 0.0;
    let alertCount = 0;
    
    // Cluster status track
    const nodeStateHistory = {
        home: { packets: 0, alerts: 0, sumRisk: 0 },
        factory: { packets: 0, alerts: 0, sumRisk: 0 },
        medical: { packets: 0, alerts: 0, sumRisk: 0 },
        grid: { packets: 0, alerts: 0, sumRisk: 0 }
    };

    // Benchmark records fetched from API
    let benchmarkData = [];

    // -------------------------------------------------
    // PRESET DATA DEFINITIONS
    // -------------------------------------------------
    const presets = {
        bot_iot: {
            "Normal Web Call": {
                seq: 15320, stddev: 0.012, N_IN_Conn_P_Src: 12, N_IN_Conn_P_Dst: 15,
                duration: 1.5, pkts: 12, bytes: 960, srate: 8.0, drate: 10.0,
                proto: "tcp", state: "CON"
            },
            "UDP Flood DDoS": {
                seq: 48922, stddev: 3.421, N_IN_Conn_P_Src: 98, N_IN_Conn_P_Dst: 98,
                duration: 0.12, pkts: 8500, bytes: 680000, srate: 70830.0, drate: 0.0,
                proto: "udp", state: "INT"
            },
            "TCP Syn DoS": {
                seq: 62001, stddev: 2.11, N_IN_Conn_P_Src: 85, N_IN_Conn_P_Dst: 85,
                duration: 0.85, pkts: 4200, bytes: 336000, srate: 4941.0, drate: 0.0,
                proto: "tcp", state: "REQ"
            },
            "Network Probe": {
                seq: 98112, stddev: 0.54, N_IN_Conn_P_Src: 300, N_IN_Conn_P_Dst: 90,
                duration: 12.4, pkts: 50, bytes: 4000, srate: 4.03, drate: 1.2,
                proto: "tcp", state: "FIN"
            }
        },
        cic_ids: {
            "HTTP Traffic": {
                flow_duration: 3500000, total_fwd_packets: 4, total_bwd_packets: 5,
                fwd_pkt_len_max: 220, fwd_pkt_len_min: 0, bwd_pkt_len_max: 340,
                bwd_pkt_len_min: 0, flow_bytes_s: 160.0, flow_pkts_s: 2.57,
                flow_iat_mean: 437500.0, flow_iat_std: 12000.0, active_mean: 5000.0,
                idle_mean: 1000000.0
            },
            "Slowloris DoS": {
                flow_duration: 95000000, total_fwd_packets: 12, total_bwd_packets: 0,
                fwd_pkt_len_max: 0, fwd_pkt_len_min: 0, bwd_pkt_len_max: 0,
                bwd_pkt_len_min: 0, flow_bytes_s: 0.0, flow_pkts_s: 0.126,
                flow_iat_mean: 7916000.0, flow_iat_std: 45000.0, active_mean: 1000.0,
                idle_mean: 90000000.0
            },
            "Portscan Probe": {
                flow_duration: 48000, total_fwd_packets: 2, total_bwd_packets: 2,
                fwd_pkt_len_max: 0, fwd_pkt_len_min: 0, bwd_pkt_len_max: 0,
                bwd_pkt_len_min: 0, flow_bytes_s: 0.0, flow_pkts_s: 83.3,
                flow_iat_mean: 16000.0, flow_iat_std: 400.0, active_mean: 0.0,
                idle_mean: 0.0
            }
        },
        kddcup99: {
            "Normal HTTP Web": {
                duration: 0, protocol_type: "tcp", service: "http", flag: "SF",
                src_bytes: 215, dst_bytes: 45076, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 1, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 1, srv_count: 1, serror_rate: 0.0,
                srv_serror_rate: 0.0, rerror_rate: 0.0, srv_rerror_rate: 0.0,
                same_srv_rate: 1.0, diff_srv_rate: 0.0, srv_diff_host_rate: 0.0,
                dst_host_count: 0, dst_host_srv_count: 0, dst_host_same_srv_rate: 1.0,
                dst_host_diff_srv_rate: 0.0, dst_host_same_src_port_rate: 1.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 0.0,
                dst_host_srv_serror_rate: 0.0, dst_host_rerror_rate: 0.0,
                dst_host_srv_rerror_rate: 0.0
            },
            "Neptune DDoS Flood": {
                duration: 0, protocol_type: "tcp", service: "private", flag: "S0",
                src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 0, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 229, srv_count: 10, serror_rate: 1.0,
                srv_serror_rate: 1.0, rerror_rate: 0.0, srv_rerror_rate: 0.0,
                same_srv_rate: 0.04, diff_srv_rate: 0.06, srv_diff_host_rate: 0.0,
                dst_host_count: 255, dst_host_srv_count: 10, dst_host_same_srv_rate: 0.04,
                dst_host_diff_srv_rate: 0.06, dst_host_same_src_port_rate: 0.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 1.0,
                dst_host_srv_serror_rate: 1.0, dst_host_rerror_rate: 0.0,
                dst_host_srv_rerror_rate: 0.0
            },
            "Port Scan Attempt": {
                duration: 2, protocol_type: "tcp", service: "private", flag: "SF",
                src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 0, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 1, srv_count: 1, serror_rate: 0.0,
                srv_serror_rate: 0.0, rerror_rate: 1.0, srv_rerror_rate: 1.0,
                same_srv_rate: 1.0, diff_srv_rate: 0.0, srv_diff_host_rate: 0.0,
                dst_host_count: 255, dst_host_srv_count: 1, dst_host_same_srv_rate: 0.0,
                dst_host_diff_srv_rate: 1.0, dst_host_same_src_port_rate: 1.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 0.0,
                dst_host_srv_serror_rate: 0.0, dst_host_rerror_rate: 1.0,
                dst_host_srv_rerror_rate: 1.0
            }
        },
        nsl_kdd: {
            "Normal HTTP Web": {
                duration: 0, protocol_type: "tcp", service: "http", flag: "SF",
                src_bytes: 215, dst_bytes: 45076, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 1, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 1, srv_count: 1, serror_rate: 0.0,
                srv_serror_rate: 0.0, rerror_rate: 0.0, srv_rerror_rate: 0.0,
                same_srv_rate: 1.0, diff_srv_rate: 0.0, srv_diff_host_rate: 0.0,
                dst_host_count: 0, dst_host_srv_count: 0, dst_host_same_srv_rate: 1.0,
                dst_host_diff_srv_rate: 0.0, dst_host_same_src_port_rate: 1.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 0.0,
                dst_host_srv_serror_rate: 0.0, dst_host_rerror_rate: 0.0,
                dst_host_srv_rerror_rate: 0.0
            },
            "Neptune DDoS Flood": {
                duration: 0, protocol_type: "tcp", service: "private", flag: "S0",
                src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 0, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 229, srv_count: 10, serror_rate: 1.0,
                srv_serror_rate: 1.0, rerror_rate: 0.0, srv_rerror_rate: 0.0,
                same_srv_rate: 0.04, diff_srv_rate: 0.06, srv_diff_host_rate: 0.0,
                dst_host_count: 255, dst_host_srv_count: 10, dst_host_same_srv_rate: 0.04,
                dst_host_diff_srv_rate: 0.06, dst_host_same_src_port_rate: 0.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 1.0,
                dst_host_srv_serror_rate: 1.0, dst_host_rerror_rate: 0.0,
                dst_host_srv_rerror_rate: 0.0
            },
            "Port Scan Attempt": {
                duration: 2, protocol_type: "tcp", service: "private", flag: "SF",
                src_bytes: 0, dst_bytes: 0, land: 0, wrong_fragment: 0, urgent: 0,
                hot: 0, num_failed_logins: 0, logged_in: 0, num_compromised: 0,
                root_shell: 0, su_attempted: 0, num_root: 0, num_file_creations: 0,
                num_shells: 0, num_access_files: 0, num_outbound_cmds: 0, is_host_login: 0,
                is_guest_login: 0, count: 1, srv_count: 1, serror_rate: 0.0,
                srv_serror_rate: 0.0, rerror_rate: 1.0, srv_rerror_rate: 1.0,
                same_srv_rate: 1.0, diff_srv_rate: 0.0, srv_diff_host_rate: 0.0,
                dst_host_count: 255, dst_host_srv_count: 1, dst_host_same_srv_rate: 0.0,
                dst_host_diff_srv_rate: 1.0, dst_host_same_src_port_rate: 1.0,
                dst_host_srv_diff_host_rate: 0.0, dst_host_serror_rate: 0.0,
                dst_host_srv_serror_rate: 0.0, dst_host_rerror_rate: 1.0,
                dst_host_srv_rerror_rate: 1.0
            }
        }
    };

    // -------------------------------------------------
    // SELECTORS & UI ELEMENTS
    // -------------------------------------------------
    const datasetSelect = document.getElementById("dataset-select");
    const modelSelect = document.getElementById("model-select");
    const scopeSelect = document.getElementById("scope-select");
    
    const btnStartStream = document.getElementById("btn-start-stream");
    const btnStopStream = document.getElementById("btn-stop-stream");
    const intervalSlider = document.getElementById("interval-slider");
    const intervalVal = document.getElementById("interval-val");
    
    const statLatency = document.getElementById("stat-latency");
    const statFps = document.getElementById("stat-fps");
    const statDetRate = document.getElementById("stat-det-rate");
    const statTotal = document.getElementById("stat-total");
    const alertCounter = document.getElementById("alert-counter");
    const sysClock = document.getElementById("sys-clock");
    
    const logsTableBody = document.getElementById("logs-table-body");
    const presetButtonsContainer = document.getElementById("preset-buttons-container");
    const dynamicFormFields = document.getElementById("dynamic-form-fields");
    const packetDiagnosticsForm = document.getElementById("packet-diagnostics-form");
    const diagnosticsResults = document.getElementById("diagnostics-results");
    
    // Benchmark controls
    const benchmarkTabs = document.querySelectorAll(".tab-btn");
    const benchmarkTableBody = document.getElementById("benchmark-table-body");

    // -------------------------------------------------
    // INITIALIZATION & TIMERS
    // -------------------------------------------------
    // Update live clock
    setInterval(() => {
        const now = new Date();
        sysClock.innerText = now.toTimeString().split(" ")[0];
    }, 1000);

    // Initialize Chart.js
    initTimelineChart();

    // Fetch historical logs on startup
    fetchLogsHistory();

    // Fetch benchmark values on startup
    fetchBenchmarks();

    // Setup Form & Presets for default selected dataset
    setupDiagnosticsForm(datasetSelect.value);

    // -------------------------------------------------
    // EVENTS LISTENERS
    // -------------------------------------------------
    datasetSelect.addEventListener("change", (e) => {
        const dataset = e.target.value;
        setupDiagnosticsForm(dataset);
        // If streaming is active, reset stream parameters or notify
        if (isStreaming) {
            pauseStreaming();
            startStreaming();
        }
    });

    modelSelect.addEventListener("change", () => {
        if (isStreaming) {
            pauseStreaming();
            startStreaming();
        }
    });

    scopeSelect.addEventListener("change", () => {
        if (isStreaming) {
            pauseStreaming();
            startStreaming();
        }
    });

    // Stream control events
    btnStartStream.addEventListener("click", () => {
        startStreaming();
    });

    btnStopStream.addEventListener("click", () => {
        pauseStreaming();
    });

    intervalSlider.addEventListener("input", (e) => {
        streamIntervalTime = parseInt(e.target.value);
        intervalVal.innerText = `${streamIntervalTime}ms`;
        if (isStreaming) {
            pauseStreaming();
            startStreaming();
        }
    });

    // Benchmark Tabs switching
    benchmarkTabs.forEach(tab => {
        tab.addEventListener("click", (e) => {
            benchmarkTabs.forEach(t => t.classList.remove("active"));
            e.target.classList.add("active");
            const dset = e.target.dataset.dataset.toUpperCase().replace("-", "_");
            renderBenchmarkTable(dset);
        });
    });

    // Handle single diagnostics submit
    packetDiagnosticsForm.addEventListener("submit", (e) => {
        e.preventDefault();
        runSingleDiagnostics();
    });

    // -------------------------------------------------
    // CORE FUNCTIONS
    // -------------------------------------------------

    // Initialise Chart.js scrolling chart
    function initTimelineChart() {
        const ctx = document.getElementById("live-timeline-chart").getContext("2d");
        
        // Define color gradient for styling
        const gradConf = ctx.createLinearGradient(0, 0, 0, 250);
        gradConf.addColorStop(0, 'rgba(0, 242, 254, 0.4)');
        gradConf.addColorStop(1, 'rgba(14, 165, 233, 0.02)');
        
        const gradLat = ctx.createLinearGradient(0, 0, 0, 250);
        gradLat.addColorStop(0, 'rgba(239, 68, 68, 0.25)');
        gradLat.addColorStop(1, 'rgba(239, 68, 68, 0.01)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Anomaly Confidence (%)',
                        data: [],
                        borderColor: '#00f2fe',
                        backgroundColor: gradConf,
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Inference Latency (ms)',
                        data: [],
                        borderColor: '#ef4444',
                        backgroundColor: gradLat,
                        borderWidth: 1.5,
                        borderDash: [5, 5],
                        fill: true,
                        tension: 0.4,
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Outfit', size: 11 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#64748b', font: { family: 'Fira Code', size: 9 } }
                    },
                    y: {
                        min: 0,
                        max: 100,
                        grid: { color: 'rgba(255, 255, 255, 0.03)' },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit' } },
                        title: { display: true, text: 'Confidence (%)', color: '#94a3b8' }
                    },
                    y1: {
                        position: 'right',
                        min: 0,
                        grid: { drawOnChartArea: false },
                        ticks: { color: '#ef4444', font: { family: 'Fira Code' } },
                        title: { display: true, text: 'Latency (ms)', color: '#ef4444' }
                    }
                }
            }
        });
    }

    // Update the timeline chart in real-time
    function updateTimelineChart(label, confValue, latencyValue) {
        chartDataBuffer.labels.push(label);
        chartDataBuffer.confidence.push(confValue * 100);
        chartDataBuffer.latency.push(latencyValue * 1000); // sec to ms

        // Maintain scrolling buffer of 30 records
        if (chartDataBuffer.labels.length > 30) {
            chartDataBuffer.labels.shift();
            chartDataBuffer.confidence.shift();
            chartDataBuffer.latency.shift();
        }

        chartInstance.data.labels = chartDataBuffer.labels;
        chartInstance.data.datasets[0].data = chartDataBuffer.confidence;
        chartInstance.data.datasets[1].data = chartDataBuffer.latency;
        chartInstance.update('none'); // Update without full animation rendering for performance
    }

    // Setup the diagnostics inputs form dynamically matching dataset columns
    function setupDiagnosticsForm(dataset) {
        const datasetPresets = presets[dataset] || {};
        
        // Render preset buttons
        presetButtonsContainer.innerHTML = "";
        Object.keys(datasetPresets).forEach(name => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "btn btn-secondary btn-mini";
            btn.innerText = name;
            btn.addEventListener("click", () => {
                loadPresetValues(dataset, name);
            });
            presetButtonsContainer.appendChild(btn);
        });

        // Gather feature keys based on preset keys
        const samplePreset = Object.values(datasetPresets)[0];
        if (!samplePreset) return;

        // Render inputs grid
        dynamicFormFields.innerHTML = "";
        Object.keys(samplePreset).forEach(key => {
            const fieldDiv = document.createElement("div");
            fieldDiv.className = "diag-field";
            
            const label = document.createElement("label");
            label.innerText = key;
            label.title = key;
            
            let inputElement;
            const value = samplePreset[key];
            
            // Check if categorical dropdown
            if (key === "proto" || key === "protocol_type") {
                inputElement = document.createElement("select");
                ["tcp", "udp", "icmp"].forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.text = opt;
                    if (opt === value) o.selected = true;
                    inputElement.appendChild(o);
                });
            } else if (key === "state") {
                inputElement = document.createElement("select");
                ["CON", "INT", "FIN", "URP", "REQ"].forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.text = opt;
                    if (opt === value) o.selected = true;
                    inputElement.appendChild(o);
                });
            } else if (key === "service") {
                inputElement = document.createElement("select");
                ["http", "private", "other", "ftp", "smtp", "dns"].forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.text = opt;
                    if (opt === value) o.selected = true;
                    inputElement.appendChild(o);
                });
            } else if (key === "flag") {
                inputElement = document.createElement("select");
                ["SF", "S0", "REJ", "RSTR"].forEach(opt => {
                    const o = document.createElement("option");
                    o.value = opt;
                    o.text = opt;
                    if (opt === value) o.selected = true;
                    inputElement.appendChild(o);
                });
            } else {
                inputElement = document.createElement("input");
                inputElement.type = "number";
                inputElement.step = "any";
                inputElement.value = value;
            }
            
            inputElement.id = `input-field-${key}`;
            inputElement.name = key;
            
            fieldDiv.appendChild(label);
            fieldDiv.appendChild(inputElement);
            dynamicFormFields.appendChild(fieldDiv);
        });
    }

    // Load values from a specific preset
    function loadPresetValues(dataset, presetName) {
        const preset = presets[dataset]?.[presetName];
        if (!preset) return;
        Object.keys(preset).forEach(key => {
            const input = document.getElementById(`input-field-${key}`);
            if (input) {
                input.value = preset[key];
            }
        });
    }

    // Fetch sqlite database logs
    function fetchLogsHistory() {
        fetch("/api/logs")
            .then(res => res.json())
            .then(logs => {
                logsTableBody.innerHTML = "";
                // Render last 30 logs in list
                const sliceLogs = logs.slice(-30).reverse();
                sliceLogs.forEach(log => {
                    appendLogRow(log, false);
                });
            })
            .catch(err => console.error("Error fetching threat logs:", err));
    }

    // Load benchmark values from API
    function fetchBenchmarks() {
        fetch("/api/benchmark")
            .then(res => res.json())
            .then(data => {
                benchmarkData = data;
                // Render default active tab dataset benchmark
                const activeTab = document.querySelector(".tab-btn.active");
                if (activeTab) {
                    const dset = activeTab.dataset.dataset.toUpperCase().replace("-", "_");
                    renderBenchmarkTable(dset);
                }
            })
            .catch(err => console.error("Error fetching benchmarks:", err));
    }

    // Render benchmark table according to active dataset tab
    function renderBenchmarkTable(datasetCode) {
        benchmarkTableBody.innerHTML = "";
        
        // Map UI code to backend code
        let targetDataset = datasetCode;
        if (datasetCode === "NSL_KDD") targetDataset = "NSLKDD";
        
        const filtered = benchmarkData.filter(b => b.dataset.toUpperCase() === targetDataset);
        
        filtered.forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td class="bold text-accent">${row.model}</td>
                <td><span class="badge ${row.mode === 'binary' ? 'badge-normal' : 'badge-threat'}">${row.mode}</span></td>
                <td class="bold">${(row.accuracy * 100).toFixed(2)}%</td>
                <td>${(row.precision * 100).toFixed(2)}%</td>
                <td>${(row.recall * 100).toFixed(2)}%</td>
                <td>${(row.f1_score * 100).toFixed(2)}%</td>
                <td>${row.training_time.toFixed(2)}s</td>
            `;
            benchmarkTableBody.appendChild(tr);
        });
    }

    // Append a log entry row to logs panel
    function appendLogRow(log, highlight = true) {
        const tr = document.createElement("tr");
        const isNormal = log.predicted_class.toLowerCase() === "normal";
        const dateStr = log.timestamp.split(" ")[1] || log.timestamp;
        
        tr.innerHTML = `
            <td>${dateStr}</td>
            <td class="text-muted">${log.dataset}</td>
            <td><span class="badge ${isNormal ? 'badge-normal' : 'badge-threat'}">${log.predicted_class}</span></td>
            <td class="bold">${(log.confidence_score * 100).toFixed(1)}%</td>
        `;

        if (highlight) {
            tr.className = isNormal ? "log-row-normal-new" : "log-row-new";
            logsTableBody.insertBefore(tr, logsTableBody.firstChild);
            
            // Limit to 50 items in view
            if (logsTableBody.children.length > 50) {
                logsTableBody.removeChild(logsTableBody.lastChild);
            }
        } else {
            logsTableBody.appendChild(tr);
        }
    }

    // -------------------------------------------------
    // SIMULATED STREAM LOOP CONTROLLER
    // -------------------------------------------------
    function startStreaming() {
        if (isStreaming) return;
        isStreaming = true;
        btnStartStream.disabled = true;
        btnStopStream.disabled = false;
        
        // Start streaming fetch interval loop
        streamIntervalId = setInterval(fetchStreamPacket, streamIntervalTime);
        
        // Update header visual dot
        const dot = document.querySelector(".header-status-panel .status-dot");
        if (dot) {
            dot.classList.remove("green");
            dot.classList.add("red-text");
            document.querySelector(".header-status-panel .status-label").innerText = "ENGINE: ACTIVE STREAM";
        }
    }

    function pauseStreaming() {
        if (!isStreaming) return;
        isStreaming = false;
        btnStartStream.disabled = false;
        btnStopStream.disabled = true;
        
        if (streamIntervalId) {
            clearInterval(streamIntervalId);
            streamIntervalId = null;
        }
        
        const dot = document.querySelector(".header-status-panel .status-dot");
        if (dot) {
            dot.classList.remove("red-text");
            dot.classList.add("green");
            document.querySelector(".header-status-panel .status-label").innerText = "ENGINE: IDLE";
        }
    }

    // Pull simulated packet predictions from backend API
    function fetchStreamPacket() {
        const dataset = datasetSelect.value;
        const model = modelSelect.value;
        const scope = scopeSelect.value;

        const url = `/api/stream?dataset=${dataset}&model_type=${model}&scope=${scope}`;
        
        fetch(url)
            .then(res => {
                if (!res.ok) throw new Error("API server packet fetch failed");
                return res.json();
            })
            .then(data => {
                if (!data.success) return;
                
                // 1. Process statistics
                totalPackets++;
                const isNormal = data.predicted_class.toLowerCase() === "normal";
                if (!isNormal) {
                    totalThreats++;
                    alertCount++;
                    alertCounter.innerText = alertCount;
                }
                sumLatency += data.latency;

                // Update UI text panels
                statTotal.innerText = totalPackets;
                statLatency.innerText = `${(data.latency * 1000).toFixed(1)} ms`;
                statFps.innerText = `${(1000 / streamIntervalTime).toFixed(0)} p/s`;
                
                const detectionRate = (totalThreats / totalPackets) * 100;
                statDetRate.innerText = `${detectionRate.toFixed(1)}%`;

                // 2. Append new record row
                const logObj = {
                    timestamp: new Date().toLocaleTimeString(),
                    dataset: dataset.toUpperCase(),
                    predicted_class: data.predicted_class,
                    confidence_score: data.confidence,
                    execution_time: data.latency
                };
                appendLogRow(logObj, true);

                // 3. Update active Chart timeline
                updateTimelineChart(totalPackets, data.confidence, data.latency);

                // 4. Update Node topology cards dynamically
                updateNodeTopology(data.target_cluster, data.predicted_class, data.confidence);
            })
            .catch(err => {
                console.error("Stream connection failed:", err);
                pauseStreaming();
            });
    }

    // Handles the topology card warning indicator flash states
    function updateNodeTopology(cluster, predictedClass, confidence) {
        const cardId = `node-${cluster}`;
        const card = document.getElementById(cardId);
        if (!card) return;

        // Increment nodes specific counters
        const history = nodeStateHistory[cluster];
        history.packets++;
        
        const isNormal = predictedClass.toLowerCase() === "normal";
        if (!isNormal) {
            history.alerts++;
            history.sumRisk += confidence;
        }

        // Calculate aggregate risk factor percentage
        const riskFactor = history.packets > 0 ? (history.alerts / history.packets) * 100 : 0.0;
        
        // Update nodes labels
        document.getElementById(`count-${cluster}`).innerText = history.packets;
        document.getElementById(`risk-${cluster}`).innerText = `${riskFactor.toFixed(1)}%`;

        const badge = card.querySelector(".node-status-badge");
        
        // Set visual warning classifications
        card.classList.remove("green-glow", "yellow-warning", "red-critical");
        
        if (isNormal) {
            card.classList.add("green-glow");
            badge.innerText = "SECURE";
        } else {
            // High confidence attack is critical, lower confidence is warning
            if (confidence > 0.85) {
                card.classList.add("red-critical");
                badge.innerText = `CRITICAL: ${predictedClass.toUpperCase()}`;
                
                // Play subtle beep sound or alert trigger if necessary (can expand here)
            } else {
                card.classList.add("yellow-warning");
                badge.innerText = `WARNING: ${predictedClass.toUpperCase()}`;
            }

            // Automatically reset the red pulse after 1.5 seconds back to normal or moderate warning
            setTimeout(() => {
                card.classList.remove("red-critical", "yellow-warning");
                card.classList.add("green-glow");
                badge.innerText = "SECURE";
            }, 1800);
        }
    }

    // -------------------------------------------------
    // MANUAL SINGLE DIAGNOSTICS SUBMITTER
    // -------------------------------------------------
    function runSingleDiagnostics() {
        const dataset = datasetSelect.value;
        const model = modelSelect.value;
        const scope = scopeSelect.value;
        
        // Gather features form elements values
        const inputs = dynamicFormFields.querySelectorAll("input, select");
        const features = {};
        
        inputs.forEach(input => {
            const name = input.name;
            const val = input.value;
            // Parse float if number, else keep string for mappings
            if (input.type === "number") {
                features[name] = parseFloat(val);
            } else {
                features[name] = val;
            }
        });

        // Trigger diagnostic card state loading
        const placeholder = diagnosticsResults.querySelector(".diag-placeholder");
        const content = document.getElementById("diag-results-content");
        
        placeholder.innerHTML = `
            <i class="fa-solid fa-rotate fa-spin big-shield-icon text-accent"></i>
            <h3>Analyzing Threat Vector...</h3>
            <p>Running forward pass calculation through pure NumPy weight matrices.</p>
        `;
        placeholder.classList.remove("hidden");
        content.classList.add("hidden");

        const payload = {
            dataset: dataset,
            model_type: model,
            scope: scope,
            features: features
        };

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
        .then(res => {
            if (!res.ok) return res.json().then(e => { throw new Error(e.error || "Inference failed") });
            return res.json();
        })
        .then(data => {
            if (!data.success) throw new Error("Inference call failed");

            // Complete loading state
            placeholder.classList.add("hidden");
            content.classList.remove("hidden");

            const isNormal = data.predicted_class.toLowerCase() === "normal";
            const color = isNormal ? "var(--success-color)" : "var(--danger-color)";
            const verdictText = data.predicted_class.toUpperCase();
            
            // Dynamic ring display
            const gauge = document.getElementById("confidence-gauge");
            const confPercent = Math.round(data.confidence * 100);
            gauge.style.background = `conic-gradient(${color} ${confPercent * 3.6}deg, rgba(255, 255, 255, 0.05) 0deg)`;
            
            document.getElementById("confidence-text").innerText = `${confPercent}%`;
            document.getElementById("confidence-text").style.color = color;
            
            const verdictSpan = document.getElementById("report-verdict");
            verdictSpan.innerText = verdictText;
            verdictSpan.style.color = color;

            const lvlSpan = document.getElementById("report-level");
            if (isNormal) {
                lvlSpan.innerText = "SECURE / PATTERN NORMAL";
                lvlSpan.style.color = "var(--success-color)";
            } else {
                lvlSpan.innerText = data.confidence > 0.85 ? "HIGH SECURITY ALERT" : "MODERATE ANOMALY DETECTED";
                lvlSpan.style.color = data.confidence > 0.85 ? "var(--danger-color)" : "var(--warning-color)";
            }
            
            document.getElementById("report-latency").innerText = `${(data.latency * 1000).toFixed(2)} ms`;
            
            // Re-fetch SQLite logs to sync records
            fetchLogsHistory();
        })
        .catch(err => {
            console.error("Diagnostic failure:", err);
            placeholder.innerHTML = `
                <i class="fa-solid fa-triangle-exclamation big-shield-icon red-text"></i>
                <h3 class="red-text">Inspection Error</h3>
                <p>${err.message}</p>
                <button type="button" class="btn btn-secondary btn-mini" id="btn-diag-reset" style="margin-top: 10px;">Reset</button>
            `;
            placeholder.classList.remove("hidden");
            content.classList.add("hidden");
            
            document.getElementById("btn-diag-reset").addEventListener("click", () => {
                setupDiagnosticsForm(dataset);
            });
        });
    }
});
