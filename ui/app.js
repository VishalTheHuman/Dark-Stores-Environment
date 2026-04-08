const WS_URL = "ws://" + window.location.host + "/ws/infer";
let ws = null;

// DOM Elements
const elTick = document.getElementById('val-tick');
const elTicksRem = document.getElementById('val-ticks-remaining');
const elScore = document.getElementById('val-score');
const elText = document.getElementById('val-text');
const elMessageBox = document.getElementById('sys-message-box');

const whGrid = document.getElementById('warehouse-grid');
const cityGrid = document.getElementById('city-grid');

const listHolding = document.getElementById('list-holding');
const listPending = document.getElementById('list-pending');
const listPacked = document.getElementById('list-packed');
const listDeliveries = document.getElementById('list-deliveries');
const listRiders = document.getElementById('list-riders');

const badgeHolding = document.getElementById('badge-holding');
const badgePending = document.getElementById('badge-pending');
const badgePacked = document.getElementById('badge-packed');
const badgeDeliveries = document.getElementById('badge-deliveries');

const agentForm = document.getElementById('agent-config-form');
const btnRun = document.getElementById('btn-run');
const btnStop = document.getElementById('btn-stop');
const llmLog = document.getElementById('llm-log');

// Initialize Grids
function initGrids() {
    whGrid.innerHTML = '';
    for (let r = 0; r < 10; r++) {
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.id = `w-${r}-${c}`;
            cell.dataset.r = r;
            cell.dataset.c = c;

            if (r === 0 && c === 0) {
                cell.classList.add('cell-bg-packing');
                cell.title = "Packing Station";
            }
            whGrid.appendChild(cell);
        }
    }

    cityGrid.innerHTML = '';
    for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
            const cell = document.createElement('div');
            cell.className = 'grid-cell';
            cell.id = `c-${r}-${c}`;

            if (r === 0 && c === 0) {
                cell.classList.add('cell-bg-packing');
                cell.title = "Dispatch Hub";
            }
            cityGrid.appendChild(cell);
        }
    }
}

function appendLog(message) {
    const div = document.createElement('div');
    div.className = 'log-entry';
    div.innerText = message;

    if (message.includes('[START]')) div.classList.add('log-start');
    else if (message.includes('[AGENT THINKING]')) div.classList.add('log-think');
    else if (message.includes('[ACTION]')) div.classList.add('log-action');
    else if (message.includes('[ERROR]')) div.classList.add('log-error');
    else if (message.includes('[END]')) div.classList.add('log-end');

    llmLog.appendChild(div);
    llmLog.scrollTop = llmLog.scrollHeight;
}

function render(obs) {
    if (!obs) return;

    // Header
    elTick.innerText = obs.tick || 0;
    elTicksRem.innerText = `/ ${(obs.ticks_remaining || 0) + (obs.tick || 0)} limit`;
    elScore.innerText = (obs.cumulative_reward || 0).toFixed(2);

    const msg = obs.error ? `ERROR: ${obs.error}` : (obs.text || 'OK');
    elText.innerText = msg;

    if (obs.error) {
        elMessageBox.style.backgroundColor = '#FFEBEE';
        elMessageBox.style.borderLeftColor = '#D32F2F';
        elText.style.color = '#C62828';
    } else if ((obs.text || '').includes('URGENT') || (obs.text || '').includes('STOCKOUT')) {
        elMessageBox.style.backgroundColor = '#FFF0F3'; // pink urgent
        elMessageBox.style.borderLeftColor = '#FF8FA3';
        elText.style.color = '#E91E63';
    } else {
        elMessageBox.style.backgroundColor = '#E8F5E9'; // green ok
        elMessageBox.style.borderLeftColor = '#4CAF50';
        elText.style.color = '#2E7D32';
    }

    // Reset Grids visually (Emoji clearing)
    document.querySelectorAll('.grid-cell').forEach(el => {
        el.classList.remove('cell-bg-shelf', 'cell-bg-target');
        // Do not clear bg-packing
        el.innerHTML = '';
        if (el.id !== 'w-0-0' && el.id !== 'c-0-0') {
            el.title = '';
            el.style.backgroundColor = '';
        }
    });

    // Populate shelves via Hover Title & Background color
    if (obs.shelves) {
        obs.shelves.forEach(sh => {
            const el = document.getElementById(`w-${sh.row}-${sh.col}`);
            if (el) {
                el.classList.add('cell-bg-shelf');
                el.title = `Shelf: ${sh.item_name} | Stock: ${sh.stock}`;
                if (sh.stock === 0) el.style.backgroundColor = '#FFCDD2'; // red pastel empty
                else el.innerHTML = '🗄️';
            }
        });
    }

    // Picker position (overwrite any shelf emoji)
    if (obs.picker_position) {
        const pr = obs.picker_position[0];
        const pc = obs.picker_position[1];
        const pCell = document.getElementById(`w-${pr}-${pc}`);
        if (pCell) {
            pCell.innerHTML = '🤖';
            pCell.title = "Picker Robot";
        }
    }

    // City Customers
    const drawCustomer = (pos, timer) => {
        const el = document.getElementById(`c-${pos[0]}-${pos[1]}`);
        if (el) {
            el.innerHTML = '🏠';
            el.classList.add('cell-bg-target');
            el.title = `Customer (Ticks left: ${timer})`;
            if (timer < 5) el.style.backgroundColor = '#FFCDD2'; // urgent tint
        }
    };
    if (obs.pending_orders) obs.pending_orders.forEach(o => drawCustomer(o.customer_position, o.timer_ticks));
    if (obs.active_deliveries) obs.active_deliveries.forEach(o => drawCustomer(o.customer_position, o.timer_ticks));

    // City Riders
    if (obs.riders) {
        obs.riders.forEach(r => {
            const el = document.getElementById(`c-${r.position[0]}-${r.position[1]}`);
            if (el) {
                el.innerHTML = '🛵';
                el.title = `Rider: ${r.rider_id} | Status: ${r.status}`;
            }
        });
    }

    // Side Panels Rendering
    const renderList = (domEl, items, formatter) => {
        domEl.innerHTML = '';
        if (!items || items.length === 0) {
            domEl.innerHTML = '<li class="empty-msg">None</li>';
            return;
        }
        items.forEach(i => {
            const li = document.createElement('li');
            const res = formatter(i);
            li.innerHTML = res.html;
            if (res.className) li.className = res.className;
            domEl.appendChild(li);
        });
    };

    badgeHolding.innerText = `${(obs.picker_holding || []).length}/5`;
    renderList(listHolding, obs.picker_holding, i => ({ html: `🛍️ ${i}` }));

    badgePending.innerText = (obs.pending_orders || []).length;
    renderList(listPending, obs.pending_orders, o => {
        let itemsStr = o.items.map(it => o.picked_items.includes(it) ? `✓ ${it}` : `• ${it}`).join(', ');
        return {
            html: `<strong>🧾 ${o.order_id}</strong><br><span style="color:#555">${itemsStr}</span><br>⏱️ ${o.timer_ticks} left`,
            className: o.timer_ticks < 10 ? 'urgent-item' : ''
        };
    });

    badgePacked.innerText = (obs.packed_orders || []).length;
    renderList(listPacked, obs.packed_orders, id => ({ html: `📦 ${id}` }));

    badgeDeliveries.innerText = (obs.active_deliveries || []).length;
    renderList(listDeliveries, obs.active_deliveries, d => ({
        html: `🚚 ${d.order_id} via ${d.rider_id} <br>⏱️ ${d.timer_ticks} left`
    }));

    renderList(listRiders, obs.riders, r => ({
        html: `🛵 <strong>${r.rider_id}</strong> - ${r.status} (eta: ${r.eta_ticks}t)`
    }));
}

// WS Handlers
function toggleRunningState(isRunning) {
    if (isRunning) {
        btnRun.style.display = 'none';
        btnStop.style.display = 'block';
    } else {
        btnRun.style.display = 'block';
        btnStop.style.display = 'none';
    }
}

btnRun.addEventListener('click', () => {

    if (ws) ws.close();

    const apiKey = document.getElementById('api-key').value;
    const baseUrl = document.getElementById('api-base-url').value;
    const model = document.getElementById('model-name').value;
    const task = document.getElementById('task-select').value;

    if (!apiKey) {
        alert("Oops! Please enter your API Key in the Advanced Setup area to run the AI.");
        // Open the details element so user can type it
        document.querySelector('details.advanced-panel').open = true;
        return;
    }

    llmLog.innerHTML = "";
    appendLog("Initiating AI Control sequence...");

    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        appendLog("Connected visually to simulation.");
        toggleRunningState(true);
        ws.send(JSON.stringify({
            command: "START",
            api_key: apiKey,
            base_url: baseUrl,
            model: model,
            task: task
        }));
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "state") {
                render(data.observation);
            } else if (data.type === "log") {
                appendLog(data.message);
            } else if (data.type === "done") {
                toggleRunningState(false);
                ws.close();
            }
        } catch (err) {
            console.error("Parse Error", err);
        }
    };

    ws.onerror = (error) => {
        appendLog("Connection Error.");
        toggleRunningState(false);
    };

    ws.onclose = () => {
        appendLog("Simulation Ended.");
        toggleRunningState(false);
    };
});

btnStop.addEventListener('click', () => {
    if (ws) {
        ws.close();
        ws = null;
    }
    toggleRunningState(false);
});

// Init
initGrids();
