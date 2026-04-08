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

css = """
.visuals-container { display: flex; flex-direction: row; gap: 20px; align-items: flex-start; justify-content: center; margin-top: 20px;}
.map-panel { background: #FFFFFF; border: 1px solid #EBE4D5; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); padding: 20px; flex: none; align-items: center; display: flex; flex-direction: column;}
.map-panel h2 { font-family: 'Outfit', sans-serif; font-size: 1.2rem; margin:0 0 5px 0;}
.subtitle { font-family: 'Roboto Mono', monospace; font-size: 0.8rem; color: #777; margin-bottom: 15px;}
.grid-container { display: grid; gap: 3px; background: #F0EDE5; padding: 6px; border-radius: 12px;}
.warehouse { grid-template-columns: repeat(8, 42px); grid-template-rows: repeat(10, 42px); }
.city { grid-template-columns: repeat(8, 42px); grid-template-rows: repeat(8, 42px); }
.grid-cell { width: 42px; height: 42px; background: #FFFFFF; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 1.6rem; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.02);}
.stat-box { display: flex; flex-direction: column; align-items: center;}
.stat-label { font-size: 0.8rem; color: #777; text-transform: uppercase; letter-spacing: 1px;}
.stat-value { font-size: 2rem; font-weight: bold; }
.info-section h3 { border-bottom: 1px solid #eee; margin-bottom: 10px; font-size: 1rem; }
.dashboard-container { background: #FFFFFF; border: 1px solid #EBE4D5; border-radius: 12px; padding: 20px; height: 100%; }
.item-list { list-style: none; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.item-list li { background: #FAFAFA; padding: 8px; border-radius: 6px; font-size: 0.85rem; border-left: 3px solid #A2D2FF; }
"""

def build_warehouse_html(obs: dict, picker_pos=None) -> str:
    shelves = {(s['row'], s['col']): s for s in obs.get('shelves', [])}
    pr, pc = picker_pos if picker_pos else obs.get('picker_position', (0,7))
    
    html = '<div class="map-panel"><h2>🏬 Warehouse Floor</h2><span class="subtitle">Robot Picker & Shelves</span><div class="grid-container warehouse">'
    for r in range(10):
        for c in range(8):
            bg = ''
            interior = ''
            if r==0 and c==0:
                bg = 'background-color: #FFB3C6;'
            if (r,c) in shelves:
                s = shelves[(r,c)]
                bg = 'background-color: #E2E8F0;'
                emoji = ITEM_EMOJIS.get(s['item_name'].lower(), '📦')
                if s['stock'] == 0:
                    bg = 'background-color: #FFCDD2;'
                else:
                    interior = emoji
                    
            if r==pr and c==pc:
                interior = '🤖'

            html += f'<div class="grid-cell" style="{bg}" title="Row:{r} Col:{c}">{interior}</div>'
    html += '</div></div>'
    return html

def build_city_html(obs: dict) -> str:
    html = '<div class="map-panel"><h2>🏙️ City Area</h2><span class="subtitle">Delivery Riders & Customers</span><div class="grid-container city">'
    
    c_map = {tuple(o['customer_position']): o for o in obs.get('pending_orders',[]) + obs.get('active_deliveries',[])}
    r_map = {tuple(r['position']): r for r in obs.get('riders',[])}

    for r in range(8):
        for c in range(8):
            bg = ''
            interior = ''
            if r==0 and c==0:
                bg = 'background-color: #FFB3C6;'
            
            if (r,c) in c_map:
                interior = '🏠'
                time_left = c_map[(r,c)].get('timer_ticks', 10)
                if time_left < 5:
                    bg = 'background-color: #FFCDD2;'
                else:
                    bg = 'background-color: #FFC8A2;'
                    
            if (r,c) in r_map:
                interior = '🛵'
                
            html += f'<div class="grid-cell" style="{bg}">{interior}</div>'
    html += '</div></div>'
    return html

def build_info_html(obs: dict, status_text: str, log_stream: str) -> str:
    holding = obs.get('picker_holding', [])
    pending = obs.get('pending_orders', [])
    packed = obs.get('packed_orders', [])
    delivering = obs.get('active_deliveries', [])
    riders = obs.get('riders', [])
    
    html = f"""
    <div class="dashboard-container">
        <div class="info-section">
            <h3>System Status</h3>
            <div style="font-weight:bold; color:{'#E91E63' if ('ERROR' in status_text or 'URGENT' in status_text) else '#2E7D32'};">{status_text}</div>
        </div>
        <div class="info-section" style="margin-top:15px;">
            <h3>🤖 Inventory ({len(holding)}/5)</h3>
            <ul class="item-list">"""
    
    if not holding:
        html += '<li style="border-left-color:#ccc">None</li>'
    else:
        for i in holding:
            html += f"<li>{ITEM_EMOJIS.get(i.lower(), '🛍️')} {i}</li>"
            
    html += f"""</ul>
        </div>
        <div class="info-section" style="margin-top:15px;">
            <h3>📝 Pending Orders ({len(pending)})</h3>
            <ul class="item-list">"""
            
    if not pending:
        html += '<li style="border-left-color:#ccc">None</li>'
    else:
        for o in pending:
            req = ', '.join([f"✓ {i}" if i in o['picked_items'] else f"• {i}" for i in o['items']])
            col = '#FFB3C6' if o.get('timer_ticks',10) < 10 else '#A2D2FF'
            html += f"<li style='border-left-color:{col}'><strong>🧾 {o['order_id']}</strong><br>{req}<br>⏱️ {o.get('timer_ticks', '?')} left</li>"
            
    html += f"""</ul>
        </div>
        <div class="info-section" style="margin-top:15px;">
            <h3>🛵 Riders</h3>
            <ul class="item-list">"""
            
    if not riders:
        html += '<li style="border-left-color:#ccc">None</li>'
    else:
        for r in riders:
            html += f"<li>🛵 <strong>{r['rider_id']}</strong> - {r['status']}</li>"
            
    html += "</ul></div></div>"
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

def run_gradio_inference(task, base_url, model, api_key):
    if not api_key:
        yield gr.update(), gr.update(), gr.update(), gr.update(), "Please enter an API Key.", "[ERROR] Missing API Key"
        return
        
    client = OpenAI(base_url=base_url, api_key=api_key)
    inference.MODEL_NAME = model
    env = DarkStoreEnvironment()
    
    try:
        obs_obj = env.reset(task=task)
        obs = obs_obj.dict()
        last_error = obs_obj.error
    except Exception as e:
        yield gr.update(), gr.update(), gr.update(), gr.update(), f"Initialization Error: {e}", f"[ERROR] {e}"
        return

    picker_pos = obs.get('picker_position', (0,7))
    log_text = f"[START] task={task} model={model}\n"
    
    tick_budget = obs.get('tick', 0) + obs.get('ticks_remaining', 0)
    score = 0.0

    # Initial yield
    sys_status = "Idle"
    yield (
        build_warehouse_html(obs, picker_pos),
        build_city_html(obs),
        build_info_html(obs, sys_status, log_text),
        gr.update(value=log_text),
        f"{obs.get('tick',0)} / {tick_budget}",
        f"{obs.get('cumulative_reward',0):.2f}"
    )

    for step in range(1, tick_budget + 2):
        if obs_obj.done:
            break
            
        log_text += f"⏳ [AGENT THINKING...] Step {step}\n"
        sys_status = "Agent Thinking..."
        yield (
            gr.update(), gr.update(),
            build_info_html(obs, sys_status, log_text),
            gr.update(value=log_text),
            gr.update(), gr.update()
        )
        
        action = inference.get_llm_action(client, obs_obj.text, last_error)
        action_str = inference.action_to_str(action)
        log_text += f"🚀 [ACTION] {action_str}\n"

        obs_obj = env.step(action)
        obs = obs_obj.dict()
        last_error = obs_obj.error

        if last_error:
            log_text += f"❌ [ERROR] {last_error}\n"
            sys_status = f"ERROR: {last_error}"
        else:
            sys_status = "OK"

        new_picker_pos = obs.get('picker_position', (0,7))
        
        # Animate picker step-by-step visually in Gradio
        if new_picker_pos != picker_pos:
            path = get_manhattan_path(picker_pos, new_picker_pos)
            for intermediate_pos in path:
                yield (
                    build_warehouse_html(obs, intermediate_pos),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
                time.sleep(0.15)
            picker_pos = new_picker_pos
            
        # Yield the fully updated state including city map
        yield (
            build_warehouse_html(obs, picker_pos),
            build_city_html(obs),
            build_info_html(obs, sys_status, log_text),
            gr.update(value=log_text),
            f"{obs.get('tick',0)} / {tick_budget}",
            f"{obs.get('cumulative_reward',0):.2f}"
        )
        time.sleep(0.1)

    score = env.compute_score()
    log_text += f"\n🏁 [END] Simulation completed. Score: {score:.3f}"
    
    yield (
        build_warehouse_html(obs, picker_pos),
        build_city_html(obs),
        build_info_html(obs, "Completed", log_text),
        gr.update(value=log_text),
        f"{obs.get('tick',0)} / {tick_budget}",
        f"{score:.3f}"
    )

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.HTML("<div class='header' style='text-align:center;'><h1>🏪 Dark Store Analytics (Gradio Light)</h1></div>")
    
    with gr.Row():
        with gr.Column(elem_classes="stat-box"):
            gr.HTML("<div class='stat-label'>Time Tick</div>")
            tick_display = gr.HTML("<div class='stat-value'>0 / 0</div>")
        with gr.Column(elem_classes="stat-box"):
            gr.HTML("<div class='stat-label'>Performance Score</div>")
            score_display = gr.HTML("<div class='stat-value text-green'>0.00</div>")

    with gr.Row():
        start_btn = gr.Button("▶ Start Auto-Manager", variant="primary")
            
    with gr.Row(elem_classes="visuals-container"):
        warehouse_view = gr.HTML(build_warehouse_html({}))
        city_view = gr.HTML(build_city_html({}))
        info_view = gr.HTML(build_info_html({}, "Idle", ""))

    with gr.Accordion("⚙️ Advanced AI Setup & Logs", open=False):
        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=["single_order", "concurrent_orders", "full_operations"],
                    value="single_order",
                    label="Task"
                )
                base_url_input = gr.Textbox(value="https://api.groq.com/openai/v1", label="API Base URL")
                model_input = gr.Textbox(value="moonshotai/kimi-k2-instruct", label="Model Name")
                api_key_input = gr.Textbox(placeholder="Your secret API key", type="password", label="API Key (GROQ / HF_TOKEN)")
            
            with gr.Column():
                log_output = gr.Textbox(label="Inference Log", lines=10, max_lines=15)

    start_btn.click(
        fn=run_gradio_inference,
        inputs=[task_dropdown, base_url_input, model_input, api_key_input],
        outputs=[warehouse_view, city_view, info_view, log_output, tick_display, score_display]
    )

if __name__ == "__main__":
    demo.launch()
