import os
import time
import textwrap
import gradio as gr
from openai import OpenAI
from server.dark_store_environment import DarkStoreEnvironment
import custom_inference as inference

ITEM_EMOJIS = {
    'milk': '🥛', 'bread': '🍞', 'eggs': '🥚', 'chips': '🍟', 'curd': '🥣', 'rice': '🍚',
    'dal': '🍲', 'butter': '🧈', 'oil': '🫙', 'biscuits': '🍪', 'coke': '🥤', 'juice': '🧃'
}

# Fetch the exact custom CSS and Fonts
ui_css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui", "style.css")
original_css = open(ui_css_path, "r", encoding="utf-8").read() if os.path.exists(ui_css_path) else ""

# Overrides so Gradio's skeleton gets out of the way
css = original_css + """
.gradio-container {
    max-width: none !important;
    padding: 0 !important;
    background: var(--bg-page) !important;
}
.app-container {
    padding: 24px !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}
#component-0 { 
    display: none; /* Hide default top padding in basic gradio if any */
}
/* Revert some Gradio inputs to look like the design */
/* Remove default Gradio border/bg for inputs so our classes shine */
.action-controls > div, .action-controls > button {
    margin: 0 !important;
    box-shadow: none !important;
}

/* Fix grid layout which Gradio breaks */
.custom-grid-container {
    display: grid !important;
    gap: 3px !important;
    background: #F0EDE5 !important;
    padding: 6px !important;
    border-radius: 12px !important;
}
.warehouse {
    grid-template-columns: repeat(8, 42px) !important;
    grid-template-rows: repeat(10, 42px) !important;
}
.city {
    grid-template-columns: repeat(8, 42px) !important;
    grid-template-rows: repeat(8, 42px) !important;
}
"""

font_links = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
"""


def build_warehouse_html(obs: dict, picker_pos=None) -> str:
    shelves = {(s['row'], s['col']): s for s in obs.get('shelves', [])}
    pr, pc = picker_pos if picker_pos else obs.get('picker_position', (0,7))
    
    html = '<div class="panel map-panel"><div class="panel-header"><h2>🏬 Warehouse Floor</h2><span class="subtitle">Robot Picker & Shelves</span></div><div class="custom-grid-container warehouse" style="display: grid !important; grid-template-columns: repeat(8, 42px) !important; grid-template-rows: repeat(10, 42px) !important; gap: 3px; background: #F0EDE5; padding: 6px; border-radius: 12px;">'
    for r in range(10):
        for c in range(8):
            classes = ['grid-cell']
            interior = ''
            style = ''
            title = f"Row:{r} Col:{c}"
            if r==0 and c==0:
                classes.append('cell-bg-packing')
                title = "Packing Station"
            if (r,c) in shelves:
                s = shelves[(r,c)]
                classes.append('cell-bg-shelf')
                title = f"Shelf: {s['item_name']} | Stock: {s['stock']}"
                if s['stock'] == 0:
                    style = 'background-color: #FFCDD2;'
                else:
                    interior = ITEM_EMOJIS.get(s['item_name'].lower(), '📦')
                    
            if r==pr and c==pc:
                interior = '🤖'
                title = "Picker Robot"

            style_attr = f' style="{style}; width: 42px; height: 42px; background: #FFFFFF; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.02);"' if style else ' style="width: 42px; height: 42px; background: #FFFFFF; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.02);"'
            html += f'<div class="{" ".join(classes)}"{style_attr} title="{title}">{interior}</div>'
    html += '</div></div>'
    return html

def build_city_html(obs: dict) -> str:
    html = '<div class="panel map-panel"><div class="panel-header"><h2>🏙️ City Area</h2><span class="subtitle">Delivery Riders & Customers</span></div><div class="custom-grid-container city" style="display: grid !important; grid-template-columns: repeat(8, 42px) !important; grid-template-rows: repeat(8, 42px) !important; gap: 3px; background: #F0EDE5; padding: 6px; border-radius: 12px;">'
    
    c_map = {}
    for o in obs.get('pending_orders',[]) + obs.get('active_deliveries',[]):
        if 'customer_position' in o:
            c_map[tuple(o['customer_position'])] = o
            
    r_map = {tuple(r['position']): r for r in obs.get('riders',[])}

    for r in range(8):
        for c in range(8):
            classes = ['grid-cell']
            interior = ''
            style = ''
            title = ''
            if r==0 and c==0:
                classes.append('cell-bg-packing')
                title = "Dispatch Hub"
            
            if (r,c) in c_map:
                interior = '🏠'
                classes.append('cell-bg-target')
                time_left = c_map[(r,c)].get('timer_ticks', 10)
                title = f"Customer (Ticks left: {time_left})"
                if time_left < 5:
                    style = 'background-color: #FFCDD2;'
                    
            if (r,c) in r_map:
                interior = '🛵'
                ri = r_map[(r,c)]
                title = f"Rider: {ri['rider_id']} | Status: {ri['status']}"
                
            style_attr = f' style="{style}; width: 42px; height: 42px; background: #FFFFFF; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.02);"' if style else ' style="width: 42px; height: 42px; background: #FFFFFF; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.02);"'
            title_attr = f' title="{title}"' if title else ''
            html += f'<div class="{" ".join(classes)}"{style_attr}{title_attr}>{interior}</div>'
    html += '</div></div>'
    return html

def build_info_html(obs: dict, status_text: str) -> str:
    holding = obs.get('picker_holding', [])
    pending = obs.get('pending_orders', [])
    packed = obs.get('packed_orders', [])
    delivering = obs.get('active_deliveries', [])
    riders = obs.get('riders', [])
    sys_color = '#E65100' if ('ERROR' in status_text or 'URGENT' in status_text) else '#2E7D32'

    html = f"""
        <section class="dashboard-container panel" style="height: 100%; border: 1px solid #EBE4D5; border-radius: 12px; padding: 20px; background: #FFFFFF; min-width: 280px; width: 100%; display: block;">
            <div class="info-panel layout-scroll">
                <div class="info-section alert-box">
                    <h3>System Status</h3>
                    <div id="val-text" style="color: {sys_color};">{status_text}</div>
                </div>

            <div class="info-section">
                <h3>🤖 Robot Inventory <span class="badge">{len(holding)}/5</span></h3>
                <ul class="item-list">
    """
    if not holding:
        html += '<li class="empty-msg">Empty</li>'
    else:
        for i in holding:
            html += f"<li>{ITEM_EMOJIS.get(i.lower(), '🛍️')} {i}</li>"
            
    html += f"""
                </ul>
            </div>
            <div class="info-section">
                <h3>📝 Pending Orders <span class="badge">{len(pending)}</span></h3>
                <ul class="item-list">
    """
            
    if not pending:
        html += '<li class="empty-msg">None</li>'
    else:
        for o in pending:
            req = ', '.join([f"✓ {i}" if i in o['picked_items'] else f"• {i}" for i in o['items']])
            xtra_class = 'urgent-item' if o.get('timer_ticks',10) < 10 else ''
            html += f"<li class='{xtra_class}'><strong>🧾 {o['order_id']}</strong><br>{req}<br>⏱️ {o.get('timer_ticks', '?')} left</li>"
            
    html += f"""
                </ul>
            </div>
            <div class="info-section">
                <h3>📦 Packed & Ready <span class="badge">{len(packed)}</span></h3>
                <ul class="item-list">
    """
    if not packed:
         html += '<li class="empty-msg">None</li>'
    else:
         for o in packed:
             html += f"<li><strong>🧾 {o}</strong> Ready to dispatch</li>"
    
    html += f"""
                </ul>
            </div>
            <div class="info-section">
                <h3>🛵 Active Deliveries <span class="badge">{len(delivering)}</span></h3>
                <ul class="item-list">
    """
    if not delivering:
         html += '<li class="empty-msg">None</li>'
    else:
         for d in delivering:
             html += f"<li><strong>🧾 {d['order_id']}</strong> via Rider {d['rider_id']}</li>"

    html += f"""
                </ul>
            </div>
            <div class="info-section">
                <h3>🧍 All Riders</h3>
                <ul class="item-list">
    """
            
    if not riders:
        html += '<li class="empty-msg">None</li>'
    else:
        for r in riders:
            html += f"<li>🛵 <strong>{r['rider_id']}</strong> - {r['status']}</li>"
            
    html += "</ul></div></div></section>"
    return html

def get_manhattan_path(start, end):
    path = []
    r, c = start
    tr, tc = end
    while r != tr or c != tc:
        if r < tr: r += 1
        elif r > tr: r -= 1
        elif c < tc: c += 1
        elif c > tc: c -= 1
        path.append((r, c))
    return path

def build_tick_score_html(tick, tick_budget, score):
    return (
        f'<span class="stat-value">{tick}</span><span class="stat-sub">/ {tick_budget} remaining</span>',
        f'<span class="stat-value text-green">{score:.2f}</span>'
    )

def run_gradio_inference(task, base_url, model, api_key):
    if not api_key:
         yield (gr.update(), gr.update(), gr.update(), "Please enter an API Key.",
                *build_tick_score_html(0,0,0.0))
         return
         
    client = OpenAI(base_url=base_url, api_key=api_key)
    inference.MODEL_NAME = model
    env = DarkStoreEnvironment()
    
    try:
        obs_obj = env.reset(task=task)
        obs = obs_obj.dict()
        last_error = obs_obj.error
    except Exception as e:
        yield (gr.update(), gr.update(), build_info_html({}, f"Error: {e}"), f"[ERROR] {e}",
               *build_tick_score_html(0, 0, 0.0))
        return

    picker_pos = obs.get('picker_position', (0,7))
    log_text = f"[START] task={task} model={model}\n"
    
    tick_budget = obs.get('tick', 0) + obs.get('ticks_remaining', 0)
    score = 0.0
    sys_status = "Idle"

    yield (
        build_warehouse_html(obs, picker_pos),
        build_city_html(obs),
        build_info_html(obs, sys_status),
        gr.update(value=log_text),
        *build_tick_score_html(obs.get('tick',0), tick_budget, obs.get('cumulative_reward',0))
    )

    for step in range(1, tick_budget + 2):
        if obs_obj.done:
            break
            
        log_text += f'<div class="log-entry log-think">⏳ [AGENT THINKING...] Step {step}</div>\n'
        sys_status = "Agent Thinking..."
        yield (
            gr.update(), gr.update(),
            build_info_html(obs, sys_status),
            gr.update(value=log_text),
            gr.update(), gr.update()
        )
        
        action = inference.get_llm_action(client, obs_obj.text, last_error)
        action_str = inference.action_to_str(action)
        log_text += f'<div class="log-entry log-action">🚀 [ACTION] {action_str}</div>\n'

        obs_obj = env.step(action)
        obs = obs_obj.dict()
        last_error = obs_obj.error

        if last_error:
            log_text += f'<div class="log-entry log-error">❌ [ERROR] {last_error}</div>\n'
            sys_status = f"ERROR: {last_error}"
        else:
            sys_status = "OK"

        new_picker_pos = obs.get('picker_position', (0,7))
        if new_picker_pos != picker_pos:
            path = get_manhattan_path(picker_pos, new_picker_pos)
            for intermediate_pos in path:
                yield (
                    build_warehouse_html(obs, intermediate_pos),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
                )
                time.sleep(0.15)
            picker_pos = new_picker_pos
            
        yield (
            build_warehouse_html(obs, picker_pos),
            build_city_html(obs),
            build_info_html(obs, sys_status),
            gr.update(value=log_text),
            *build_tick_score_html(obs.get('tick',0), tick_budget, obs.get('cumulative_reward',0))
        )
        time.sleep(0.1)

    score = env.compute_score()
    log_text += f'\n<div class="log-entry log-action">🏁 [END] Simulation completed. Score: {score:.3f}</div>'
    
    yield (
        build_warehouse_html(obs, picker_pos),
        build_city_html(obs),
        build_info_html(obs, "Completed"),
        gr.update(value=log_text),
        *build_tick_score_html(obs.get('tick',0), tick_budget, score)
    )

with gr.Blocks(theme=gr.themes.Soft(), css=css, head=font_links) as demo:
    with gr.Column(elem_classes="app-container"):
        # Header
        with gr.Row(elem_classes=["header", "panel"]):
            gr.HTML('<div class="logo"><h1>🏪 Dark Store Operations</h1></div>')
            with gr.Row(elem_classes="status-board", scale=1):
                with gr.Column(elem_classes="stat-box", scale=1, min_width=100):
                    gr.HTML('<span class="stat-label">Time Tick</span>')
                    tick_display = gr.HTML('<span class="stat-value">0</span><span class="stat-sub">/ 0 remaining</span>')
                with gr.Column(elem_classes="stat-box", scale=1, min_width=150):
                    gr.HTML('<span class="stat-label">Performance Score</span>')
                    score_display = gr.HTML('<span class="stat-value text-green">0.0</span>')
            with gr.Row(elem_classes="action-controls", scale=1):
                task_dropdown = gr.Dropdown(
                    choices=["single_order", "concurrent_orders", "full_operations"],
                    value="single_order",
                    show_label=False,
                    container=False,
                    elem_classes="clean-input"
                )
                start_btn = gr.Button("▶ Start Auto-Manager", elem_classes=["btn", "btn-action"], size="lg")

        # Main Workspace
        with gr.Row(elem_classes="main-workspace"):
            # Center Visuals
            with gr.Row(elem_classes="visuals-container", scale=2):
                warehouse_view = gr.HTML(build_warehouse_html({}))
                city_view = gr.HTML(build_city_html({}))
            # Info Panel
            info_view = gr.HTML(build_info_html({}, "Idle"))

        # Advanced Settings
        with gr.Accordion("⚙️ Advanced AI Setup & Logs", open=False):
            with gr.Row(elem_classes="advanced-content"):
                with gr.Column(scale=1, elem_classes="agent-form"):
                    base_url_input = gr.Textbox(value="https://api.groq.com/openai/v1", label="API Base URL", elem_classes="clean-input")
                    model_input = gr.Textbox(value="moonshotai/kimi-k2-instruct", label="Model Name", elem_classes="clean-input")
                    api_key_input = gr.Textbox(placeholder="Your secret API key", type="password", label="API Key (GROQ / HF_TOKEN)", elem_classes="clean-input")
                with gr.Column(scale=2, elem_classes="log-area"):
                    gr.HTML("<h3>Inference Log</h3>")
                    log_output = gr.HTML('<div class="llm-log-container layout-scroll"></div>') 

    # For HTML logs, we should update log_output with the HTML wrapped content
    start_btn.click(
        fn=run_gradio_inference,
        inputs=[task_dropdown, base_url_input, model_input, api_key_input],
        outputs=[warehouse_view, city_view, info_view, log_output, tick_display, score_display]
    )

if __name__ == "__main__":
    demo.launch()
