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
                cell.classList.add('cell-packing');
                cell.innerHTML = 'PACK';
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
    if(!obs) return;

    // Header
    elTick.innerText = obs.tick || 0;
    elTicksRem.innerText = `/ ${obs.ticks_remaining + obs.tick}`;
    elScore.innerText = (obs.cumulative_reward || 0).toFixed(2);
    elText.innerText = obs.error ? `ERROR: ${obs.error}` : (obs.text || 'OK');
    
    if(obs.error) {
        elMessageBox.style.backgroundColor = 'rgba(255,0,0,0.2)';
    } else if((obs.text||'').includes('URGENT')) {
        elMessageBox.style.backgroundColor = 'rgba(255,255,0,0.2)';
    } else {
        elMessageBox.style.backgroundColor = 'transparent';
    }

    // Reset Grids visually
    document.querySelectorAll('.grid-cell').forEach(el => {
        el.classList.remove('cell-picker', 'cell-shelf', 'cell-customer', 'cell-rider');
        const r = parseInt(el.dataset.r);
        const c = parseInt(el.dataset.c);
        if(!(r===0 && c===0 && el.id.startsWith('w-'))) el.innerHTML = '';
        else el.innerHTML = 'PACK';
    });

    // Populate shelves
    if(obs.shelves) {
        obs.shelves.forEach(sh => {
            const el = document.getElementById(`w-${sh.row}-${sh.col}`);
            if(el) {
                el.classList.add('cell-shelf');
                let sc = 'stock-ok';
                if(sh.stock === 0) sc = 'stock-empty';
                else if(sh.stock < 3) sc = 'stock-low';
                el.innerHTML = `<span class="shelf-name">${sh.item_name}</span><span class="shelf-stock ${sc}">${sh.stock}</span>`;
            }
        });
    }

    // Picker
    if(obs.picker_position) {
        const pr = obs.picker_position[0];
        const pc = obs.picker_position[1];
        const pCell = document.getElementById(`w-${pr}-${pc}`);
        if(pCell) {
            pCell.classList.add('cell-picker');
            pCell.innerHTML = 'P';
        }
    }

    // City Customers
    const drawCustomer = (pos, timer, text) => {
        const el = document.getElementById(`c-${pos[0]}-${pos[1]}`);
        if(el) {
            el.classList.add('cell-customer');
            el.innerHTML = `<span>${timer}s</span>`;
        }
    };
    if(obs.pending_orders) {
        obs.pending_orders.forEach(o => drawCustomer(o.customer_position, o.timer_ticks, 'P'));
    }
    if(obs.active_deliveries) {
        obs.active_deliveries.forEach(o => drawCustomer(o.customer_position, o.timer_ticks, 'D'));
    }

    // City Riders
    if(obs.riders) {
        obs.riders.forEach(r => {
            const el = document.getElementById(`c-${r.position[0]}-${r.position[1]}`);
            if(el) {
                el.classList.add('cell-rider');
                el.innerHTML = r.rider_id;
            }
        });
    }

    // Side Panels
    const renderList = (domEl, items, formatter) => {
        domEl.innerHTML = '';
        if(!items || items.length === 0) {
            domEl.innerHTML = '<li class="empty-msg">Empty</li>';
            return;
        }
        items.forEach(i => {
            const li = document.createElement('li');
            const res = formatter(i);
            li.innerHTML = res.html;
            if(res.className) li.className = res.className;
            domEl.appendChild(li);
        });
    };

    badgeHolding.innerText = `${(obs.picker_holding||[]).length}/5`;
    renderList(listHolding, obs.picker_holding, i => ({html: i}));

    badgePending.innerText = (obs.pending_orders||[]).length;
    renderList(listPending, obs.pending_orders, o => ({
        html: `<strong>${o.order_id}</strong> <br/> Need: ${o.items.join(',')} <br/> Got: ${o.picked_items.length} | ⏱️ ${o.timer_ticks}t`,
        className: o.timer_ticks < 10 ? 'urgent-order' : ''
    }));

    badgePacked.innerText = (obs.packed_orders||[]).length;
    renderList(listPacked, obs.packed_orders, id => ({html:`📦 ${id}`}));

    badgeDeliveries.innerText = (obs.active_deliveries||[]).length;
    renderList(listDeliveries, obs.active_deliveries, d => ({
        html: `🚚 ${d.order_id} (Rider: ${d.rider_id}) | ⏱️ ${d.timer_ticks}t`
    }));

    renderList(listRiders, obs.riders, r => ({
        html: `🏍️ <strong>${r.rider_id}</strong> - ${r.status} <br/> (eta: ${r.eta_ticks}t)`
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

agentForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    if (ws) {
        ws.close();
    }

    const apiKey = document.getElementById('api-key').value;
    const baseUrl = document.getElementById('api-base-url').value;
    const model = document.getElementById('model-name').value;
    const task = document.getElementById('task-select').value;
    
    if (!apiKey) {
        alert("Please enter a valid API Key (HF Token).");
        return;
    }

    llmLog.innerHTML = "";
    appendLog("Connecting to WebSocket inference engine...");
    
    ws = new WebSocket(WS_URL);
    
    ws.onopen = () => {
        appendLog("Connected & Started Background Simulation");
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
        } catch(err) {
            console.error("Parse Error", err);
        }
    };

    ws.onerror = (error) => {
        appendLog("WebSocket Error occurred.");
        console.error(error);
        toggleRunningState(false);
    };

    ws.onclose = () => {
        appendLog("Simulation Ended / Connection Closed.");
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
