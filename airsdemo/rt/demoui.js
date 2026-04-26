// UI Controller for Prisma AIRS Red Team Demo
// Manages all user interactions and display updates

class RedTeamUI {
    constructor() {
        this.currentScanId = null;
        this.pollInterval = null;
        this.categories = [];
        this.currentStep = 0;
        this.instantDemoMode = false; // New: instant demo mode toggle
        this.demoResults = null; // Store pre-loaded demo results
    }

    async initialize() {
        // Initialize API client
        initializeAPI();

        // Load demo results data
        this.loadDemoResults();

        // Check if token is configured
        if (!DEMO_CONFIG.api.bearerToken && !DEMO_CONFIG.demo.simulateMode && !this.instantDemoMode) {
            this.promptForToken();
        }

        // Load initial data
        await this.loadTargets();
        await this.loadCategories();
        this.populateModelTypes();

        // Setup event listeners
        this.setupEventListeners();

        // Hide status card initially
        document.getElementById('statusCard').classList.add('hidden');

        // Initialize process flow at step 1
        this.updateProcessFlow(1);
    }

    loadDemoResults() {
        // Check if demo results are available
        if (typeof DEMO_SCAN_RESULTS !== 'undefined') {
            this.demoResults = DEMO_SCAN_RESULTS;
            console.log('Demo results loaded:', Object.keys(this.demoResults));
        }
    }

    toggleInstantDemo() {
        this.instantDemoMode = !this.instantDemoMode;
        const toggle = document.getElementById('demoModeToggle');
        const banner = document.getElementById('demoBanner');
        const icon = document.getElementById('demoModeIcon');
        const text = document.getElementById('demoModeText');

        if (this.instantDemoMode) {
            toggle.classList.add('active');
            banner.classList.add('active');
            icon.textContent = '✓';
            text.textContent = 'Instant Demo: ON';
            console.log('Instant demo mode ENABLED');
        } else {
            toggle.classList.remove('active');
            banner.classList.remove('active');
            icon.textContent = '⚡';
            text.textContent = 'Instant Demo Mode';
            console.log('Instant demo mode DISABLED');
        }
    }

    updateProcessFlow(step) {
        this.currentStep = step;
        const steps = document.querySelectorAll('.flow-step');
        const connectors = document.querySelectorAll('.flow-connector');

        // Update steps
        steps.forEach((stepEl, index) => {
            const stepNum = index + 1;
            if (stepNum < step) {
                stepEl.classList.add('completed');
                stepEl.classList.remove('active');
            } else if (stepNum === step) {
                stepEl.classList.add('active');
                stepEl.classList.remove('completed');
            } else {
                stepEl.classList.remove('active', 'completed');
            }
        });

        // Update connectors
        connectors.forEach((connector, index) => {
            const connectorNum = index + 1;
            if (connectorNum < step) {
                connector.classList.add('completed');
                connector.classList.remove('active');
            } else if (connectorNum === step) {
                connector.classList.add('active');
                connector.classList.remove('completed');
            } else {
                connector.classList.remove('active', 'completed');
            }
        });
    }

    promptForToken() {
        const token = prompt(
            'Please enter your Prisma AIRS API Bearer Token:\n\n' +
            'You can obtain this token from your Strata Cloud Manager.'
        );

        if (token) {
            DEMO_CONFIG.api.bearerToken = token;
            apiClient.setToken(token);
        } else if (!DEMO_CONFIG.demo.simulateMode) {
            alert('Demo mode enabled - using simulated data');
            DEMO_CONFIG.demo.simulateMode = true;
        }
    }

    async loadTargets() {
        const select = document.getElementById('targetSelect');
        select.innerHTML = '<option value="">Loading targets...</option>';

        try {
            const response = await apiClient.listTargets();
            const targets = response.data || DEMO_CONFIG.targets;

            select.innerHTML = '<option value="">Select a target...</option>';
            targets.forEach(target => {
                const option = document.createElement('option');
                option.value = target.uuid;
                option.textContent = `${target.name} (${target.target_type})`;
                option.dataset.target = JSON.stringify(target);
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load targets:', error);
            select.innerHTML = '<option value="">Error loading targets</option>';
            this.showError('Failed to load targets. Please check your configuration.');
        }
    }

    async loadCategories() {
        try {
            this.categories = await apiClient.getCategories();
            console.log('Loaded categories:', this.categories);
        } catch (error) {
            console.error('Failed to load categories:', error);
        }
    }

    populateModelTypes() {
        const select = document.getElementById('modelSelect');
        select.innerHTML = '<option value="">Select a model...</option>';

        DEMO_CONFIG.modelTypes.forEach(model => {
            const option = document.createElement('option');
            option.value = model.value;
            option.textContent = model.label;
            select.appendChild(option);
        });
    }

    setupEventListeners() {
        document.getElementById('startScanBtn').addEventListener('click', () => {
            this.startScan();
        });

        // Update process flow on target selection
        document.getElementById('targetSelect').addEventListener('change', () => {
            if (document.getElementById('targetSelect').value) {
                this.updateProcessFlow(1);
            }
        });

        // Update process flow on model selection
        document.getElementById('modelSelect').addEventListener('change', () => {
            if (document.getElementById('modelSelect').value) {
                this.updateProcessFlow(2);
            }
        });
    }

    async startScan() {
        const targetSelect = document.getElementById('targetSelect');
        const modelSelect = document.getElementById('modelSelect');
        const scanTypeSelect = document.getElementById('scanType');
        const scanNameInput = document.getElementById('scanName');

        // Validation
        if (!targetSelect.value) {
            alert('Please select a target');
            return;
        }

        if (!modelSelect.value) {
            alert('Please select a model type');
            return;
        }

        if (!scanNameInput.value.trim()) {
            alert('Please enter a scan name');
            return;
        }

        // Check if instant demo mode is active
        if (this.instantDemoMode) {
            this.runInstantDemo(targetSelect, modelSelect, scanTypeSelect, scanNameInput);
            return;
        }

        // Get target data
        const targetOption = targetSelect.options[targetSelect.selectedIndex];
        const target = JSON.parse(targetOption.dataset.target);

        // Prepare scan request
        const scanRequest = {
            name: scanNameInput.value.trim(),
            target: {
                target_id: target.uuid,
                target_type: target.target_type,
                target_metadata: {
                    model_name: modelSelect.value
                }
            },
            job_type: scanTypeSelect.value,
            job_metadata: {
                categories: this.categories.map(cat => ({
                    id: cat.id,
                    sub_categories: cat.sub_categories.map(sub => sub.id)
                })),
                ...DEMO_CONFIG.scanDefaults.metadata
            }
        };

        // Disable button and show status
        const button = document.getElementById('startScanBtn');
        button.disabled = true;
        button.textContent = 'Starting Scan...';

        try {
            // Create scan
            const scan = await apiClient.createScan(scanRequest);
            this.currentScanId = scan.uuid;

            // Update process flow to Launch step
            this.updateProcessFlow(5);

            // Show status card and hide results
            document.getElementById('statusCard').classList.remove('hidden');
            document.getElementById('resultsCard').classList.add('hidden');

            // Start polling for status
            this.startStatusPolling();

        } catch (error) {
            console.error('Failed to start scan:', error);
            this.showError('Failed to start scan: ' + error.message);
            button.disabled = false;
            button.textContent = 'Launch Red Team Scan';
        }
    }

    runInstantDemo(targetSelect, modelSelect, scanTypeSelect, scanNameInput) {
        console.log('Running instant demo mode...');

        // Get demo key based on target selection
        const targetName = targetSelect.options[targetSelect.selectedIndex].text.toLowerCase();
        let demoKey = 'demo-chatbot-scan'; // default

        if (targetName.includes('agent') || targetName.includes('travel')) {
            demoKey = 'demo-agent-scan';
        } else if (targetName.includes('secure') || targetName.includes('gpt-4')) {
            demoKey = 'demo-secure-model';
        }

        // Get the demo results
        const demoData = this.demoResults[demoKey];

        if (!demoData) {
            alert('Demo data not available for this target');
            return;
        }

        // Disable button
        const button = document.getElementById('startScanBtn');
        button.disabled = true;
        button.textContent = 'Loading Demo...';

        // Animate through all process steps quickly
        this.animateInstantDemo(demoData, scanTypeSelect.value);
    }

    animateInstantDemo(demoData, scanType) {
        // Show status card
        document.getElementById('statusCard').classList.remove('hidden');
        document.getElementById('resultsCard').classList.add('hidden');

        // Animate through steps with delays
        const steps = [
            { step: 1, delay: 0, message: 'Target selected: ' + demoData.scanInfo.target },
            { step: 2, delay: 500, message: 'Model configured: ' + demoData.scanInfo.model },
            { step: 3, delay: 1000, message: 'Scan type: ' + demoData.scanInfo.scanType },
            { step: 4, delay: 1500, message: 'Loading attack categories...' },
            { step: 5, delay: 2000, message: 'Executing red team attacks...' },
            { step: 6, delay: 3000, message: 'Analyzing ' + demoData.scanInfo.totalAttacks + ' attack scenarios...' },
            { step: 7, delay: 4000, message: 'Generating comprehensive report...' }
        ];

        steps.forEach(({ step, delay, message }) => {
            setTimeout(() => {
                this.updateProcessFlow(step);
                this.updateStatusDisplay('RUNNING', message, (step / 7) * 100);
            }, delay);
        });

        // Show results after animation
        setTimeout(() => {
            this.displayInstantDemoResults(demoData, scanType);
        }, 4500);
    }

    displayInstantDemoResults(demoData, scanType) {
        // Update process flow to complete
        this.updateProcessFlow(7);

        // Hide status, show results
        document.getElementById('statusCard').classList.add('hidden');
        document.getElementById('resultsCard').classList.remove('hidden');

        // Display metrics
        this.displayMetrics(demoData.report, scanType);

        // Display risk score
        this.displayRiskScore(demoData.report.score || demoData.report.Score);

        // Display attacks table
        if (demoData.attacks) {
            this.displayAttacksTable(demoData.attacks);
        }

        // Add demo info banner to results
        const resultsCard = document.getElementById('resultsCard');
        const existingBanner = resultsCard.querySelector('.demo-info-banner');
        if (existingBanner) existingBanner.remove();

        const demoBanner = document.createElement('div');
        demoBanner.className = 'demo-info-banner';
        demoBanner.innerHTML = `
            <div style="background: rgba(255, 184, 0, 0.1); border: 1px solid var(--warning); border-radius: 12px; padding: 16px; margin-bottom: 24px;">
                <strong style="color: var(--warning);">📊 Demo Results</strong> -
                Scan completed: ${demoData.scanInfo.duration} |
                ${demoData.scanInfo.totalAttacks || demoData.scanInfo.totalGoals} scenarios tested
            </div>
        `;
        resultsCard.insertBefore(demoBanner, resultsCard.firstChild.nextSibling);

        // Reset UI
        this.resetUI();
    }

    startStatusPolling() {
        let progressValue = 0;
        let statusMessages = [
            'Initializing attack vectors...',
            'Loading vulnerability database...',
            'Executing prompt injections...',
            'Testing jailbreak scenarios...',
            'Analyzing model responses...',
            'Detecting PII leakage...',
            'Evaluating content safety...',
            'Compiling threat report...'
        ];
        let messageIndex = 0;

        // Update process flow to Red Team Attack step (step 5)
        this.updateProcessFlow(5);

        // Update UI immediately
        this.updateStatusDisplay('RUNNING', statusMessages[0], 10);

        // Poll for actual status
        this.pollInterval = setInterval(async () => {
            try {
                const status = await apiClient.getScanStatus(this.currentScanId);

                // Calculate progress
                const progress = status.completed && status.total
                    ? Math.round((status.completed / status.total) * 100)
                    : progressValue;

                // Update message based on progress
                messageIndex = Math.min(
                    Math.floor((progress / 100) * statusMessages.length),
                    statusMessages.length - 1
                );

                this.updateStatusDisplay(
                    status.status,
                    statusMessages[messageIndex],
                    progress
                );

                // Check if scan is complete
                if (status.status === 'COMPLETED') {
                    clearInterval(this.pollInterval);
                    await this.loadResults();
                } else if (status.status === 'FAILED') {
                    clearInterval(this.pollInterval);
                    this.showError('Scan failed. Please try again.');
                    this.resetUI();
                }

                progressValue = Math.min(progressValue + 5, 95);

            } catch (error) {
                console.error('Failed to get scan status:', error);
            }
        }, DEMO_CONFIG.demo.refreshInterval);

        // Simulated progress for demo mode
        if (DEMO_CONFIG.demo.simulateMode) {
            const simulateInterval = setInterval(() => {
                progressValue += 2;
                messageIndex = Math.min(
                    Math.floor((progressValue / 100) * statusMessages.length),
                    statusMessages.length - 1
                );

                this.updateStatusDisplay('RUNNING', statusMessages[messageIndex], progressValue);

                if (progressValue >= 100) {
                    clearInterval(simulateInterval);
                    clearInterval(this.pollInterval);
                    setTimeout(() => this.loadResults(), 1000);
                }
            }, 300);
        }
    }

    updateStatusDisplay(status, message, progress) {
        const statusText = document.querySelector('.status-text');
        const statusDetail = document.querySelector('.status-detail');
        const progressBar = document.getElementById('progressBar');

        statusText.textContent = message;
        statusDetail.textContent = `Status: ${status} • Progress: ${progress}%`;
        progressBar.style.width = `${progress}%`;
    }

    async loadResults() {
        try {
            // Get report based on scan type
            const scanType = document.getElementById('scanType').value;
            let report;

            if (scanType === 'STATIC') {
                report = await apiClient.getStaticReport(this.currentScanId);
                const attacks = await apiClient.listAttacks(this.currentScanId, 0, 20);
                report.attacks = attacks.data || [];
            } else if (scanType === 'DYNAMIC') {
                report = await apiClient.getDynamicReport(this.currentScanId);
            }

            this.displayResults(report, scanType);

        } catch (error) {
            console.error('Failed to load results:', error);
            this.showError('Failed to load results: ' + error.message);
        }
    }

    displayResults(report, scanType) {
        // Update process flow to Generate Report step (step 7)
        this.updateProcessFlow(7);

        // Hide status, show results
        document.getElementById('statusCard').classList.add('hidden');
        document.getElementById('resultsCard').classList.remove('hidden');

        // Display metrics
        this.displayMetrics(report, scanType);

        // Display risk score
        this.displayRiskScore(report.score || report.Score);

        // Display attacks table (for static scans)
        if (scanType === 'STATIC' && report.attacks) {
            this.displayAttacksTable(report.attacks);
        }

        // Reset UI
        this.resetUI();
    }

    displayMetrics(report, scanType) {
        const grid = document.getElementById('metricsGrid');
        grid.innerHTML = '';

        if (scanType === 'STATIC') {
            const metrics = [
                {
                    label: 'Critical Threats',
                    value: report.severity_report?.critical || 0,
                    class: 'critical'
                },
                {
                    label: 'High Severity',
                    value: report.severity_report?.high || 0,
                    class: 'warning'
                },
                {
                    label: 'Medium Severity',
                    value: report.severity_report?.medium || 0,
                    class: 'warning'
                },
                {
                    label: 'Attack Success Rate',
                    value: `${Math.round((report.asr || 0) * 100)}%`,
                    class: report.asr > 0.2 ? 'critical' : 'success'
                }
            ];

            metrics.forEach(metric => {
                const card = document.createElement('div');
                card.className = `metric-card ${metric.class}`;
                card.innerHTML = `
                    <div class="metric-value">${metric.value}</div>
                    <div class="metric-label">${metric.label}</div>
                `;
                grid.appendChild(card);
            });
        } else if (scanType === 'DYNAMIC') {
            const metrics = [
                {
                    label: 'Total Goals',
                    value: report.total_goals || 0,
                    class: 'info'
                },
                {
                    label: 'Goals Achieved',
                    value: report.goals_achieved || 0,
                    class: 'warning'
                },
                {
                    label: 'Total Threats',
                    value: report.total_threats || 0,
                    class: 'critical'
                },
                {
                    label: 'Attack Success Rate',
                    value: `${Math.round((report.asr || 0) * 100)}%`,
                    class: report.asr > 0.2 ? 'critical' : 'success'
                }
            ];

            metrics.forEach(metric => {
                const card = document.createElement('div');
                card.className = `metric-card ${metric.class}`;
                card.innerHTML = `
                    <div class="metric-value">${metric.value}</div>
                    <div class="metric-label">${metric.label}</div>
                `;
                grid.appendChild(card);
            });
        }
    }

    displayRiskScore(score) {
        const container = document.getElementById('riskScore');

        let riskLevel = 'low';
        let riskText = 'Low Risk';
        if (score < 60) {
            riskLevel = 'high';
            riskText = 'High Risk';
        } else if (score < 80) {
            riskLevel = 'medium';
            riskText = 'Medium Risk';
        }

        container.innerHTML = `
            <div class="score-circle ${riskLevel}">
                ${score}
                <div class="score-label">${riskText}</div>
            </div>
            <div style="flex: 1;">
                <h3 style="margin-bottom: 12px;">Security Score</h3>
                <p style="color: var(--text-secondary); line-height: 1.8;">
                    The security score represents the overall resilience of your AI model
                    against adversarial attacks. A score of ${score} indicates ${riskText.toLowerCase()}
                    exposure to potential security vulnerabilities.
                </p>
            </div>
        `;
    }

    displayAttacksTable(attacks) {
        const container = document.getElementById('attacksTableContainer');

        if (!attacks || attacks.length === 0) {
            container.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 24px;">No attacks to display</p>';
            return;
        }

        const table = document.createElement('table');
        table.className = 'attacks-table';
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Attack Type</th>
                    <th>Severity</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                ${attacks.slice(0, DEMO_CONFIG.ui.maxAttacksToShow).map(attack => `
                    <tr>
                        <td>${attack.category || 'Unknown'}</td>
                        <td>${attack.sub_category || attack.attack_type || 'N/A'}</td>
                        <td>
                            <span class="severity-badge ${(attack.severity || 'low').toLowerCase()}">
                                ${attack.severity || 'Low'}
                            </span>
                        </td>
                        <td>${attack.threat ? '⚠️ Threat Detected' : '✅ Blocked'}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;

        container.innerHTML = '<h3 style="margin: 24px 0 16px;">Attack Details</h3>';
        container.appendChild(table);
    }

    showError(message) {
        const statusCard = document.getElementById('statusCard');
        statusCard.classList.remove('hidden');

        const statusContent = statusCard.querySelector('.status-content');
        statusContent.innerHTML = `
            <div class="error-message">
                <span class="error-icon">⚠️</span>
                <div>${message}</div>
            </div>
        `;
    }

    resetUI() {
        const button = document.getElementById('startScanBtn');
        button.disabled = false;
        button.textContent = 'Launch Red Team Scan';

        // Reset process flow to step 1
        this.updateProcessFlow(1);
    }
}

// Initialize UI when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const ui = new RedTeamUI();
    window.redTeamUI = ui; // Make globally accessible for toggle function
    ui.initialize();
});