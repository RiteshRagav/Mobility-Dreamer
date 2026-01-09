"""
Interactive Policy Editor - Gradio Web Interface
=================================================
Web-based GUI for drawing policy interventions on urban scenes.

Features:
- Draw bike lanes, pedestrian zones, green spaces, EV charging stations
- Real-time preview
- Color-coded intervention types
- Export as PNG masks + JSON metadata

Requirements:
    pip install gradio

Usage:
    python src/policy_gui.py --frames data/frames --out data/policy_maps
    
    Then open http://localhost:7860 in your browser
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("⚠️  Gradio not installed. Install with: pip install gradio")


# Policy intervention colors (RGB for display, BGR for saving)
POLICY_TYPES = {
    "Bike Lane": {"color": (0, 255, 0), "bgr": (0, 255, 0), "desc": "Dedicated cycling infrastructure"},
    "Pedestrian Zone": {"color": (0, 100, 255), "bgr": (255, 100, 0), "desc": "Car-free pedestrian areas"},
    "Green Space": {"color": (180, 105, 255), "bgr": (255, 105, 180), "desc": "Parks and green infrastructure"},
    "EV Charging": {"color": (0, 165, 255), "bgr": (255, 165, 0), "desc": "Electric vehicle charging stations"},
    "Bus Lane": {"color": (255, 255, 0), "bgr": (0, 255, 255), "desc": "Dedicated bus rapid transit"},
}


class PolicyEditor:
    """
    Interactive policy editor application.
    """
    
    def __init__(self, frames_dir, output_dir):
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all frames
        self.frame_files = sorted(self.frames_dir.glob("frame_*.jpg")) + sorted(self.frames_dir.glob("frame_*.png"))
        
        if len(self.frame_files) == 0:
            raise ValueError(f"No frames found in {frames_dir}")
        
        print(f"📁 Loaded {len(self.frame_files)} frames")
        
        self.current_frame_idx = 0
        self.current_policy_map = None
        self.interventions_metadata = []
    
    def get_current_frame(self):
        """Load current frame as PIL Image."""
        frame_path = self.frame_files[self.current_frame_idx]
        return Image.open(frame_path)
    
    def create_blank_policy_map(self, size):
        """Create blank policy map."""
        return Image.new('RGB', size, (0, 0, 0))
    
    def draw_bike_lane(self, image, points, width=40):
        """Draw bike lane on policy map."""
        draw = ImageDraw.Draw(image)
        color = POLICY_TYPES["Bike Lane"]["color"]
        draw.line(points, fill=color, width=width)
        return image
    
    def draw_pedestrian_zone(self, image, points):
        """Draw pedestrian zone on policy map."""
        draw = ImageDraw.Draw(image)
        color = POLICY_TYPES["Pedestrian Zone"]["color"]
        draw.polygon(points, fill=color)
        return image
    
    def draw_green_space(self, image, points):
        """Draw green space on policy map."""
        draw = ImageDraw.Draw(image)
        color = POLICY_TYPES["Green Space"]["color"]
        draw.polygon(points, fill=color)
        return image
    
    def draw_ev_station(self, image, center, radius=30):
        """Draw EV charging station marker."""
        draw = ImageDraw.Draw(image)
        color = POLICY_TYPES["EV Charging"]["color"]
        bbox = [center[0]-radius, center[1]-radius, center[0]+radius, center[1]+radius]
        draw.ellipse(bbox, fill=color)
        return image
    
    def save_policy_map(self, policy_map_pil, metadata):
        """Save policy map and metadata."""
        frame_name = self.frame_files[self.current_frame_idx].name
        
        # Save policy map as PNG
        policy_name = frame_name.replace("frame_", "policy_")
        policy_path = self.output_dir / policy_name
        
        # Convert RGB to BGR for OpenCV compatibility
        policy_np = np.array(policy_map_pil)
        policy_bgr = cv2.cvtColor(policy_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(policy_path), policy_bgr)
        
        # Save metadata as JSON
        metadata_name = frame_name.replace("frame_", "policy_").replace(".jpg", ".json").replace(".png", ".json")
        metadata_path = self.output_dir / metadata_name
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved policy map: {policy_path}")
        return policy_path


def create_gradio_interface(editor):
    """
    Create Gradio web interface for policy editing.
    
    Args:
        editor: PolicyEditor instance
    """
    
    with gr.Blocks(title="MobilityDreamer Policy Editor") as demo:
        gr.Markdown("""
        # 🚴 MobilityDreamer - Policy Intervention Editor
        
        Draw infrastructure changes on urban scenes:
        - **Green**: Bike lanes
        - **Blue**: Pedestrian zones  
        - **Pink**: Green spaces
        - **Orange**: EV charging stations
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Image canvas
                frame_display = gr.Image(label="Current Frame", type="pil")
                policy_canvas = gr.Image(label="Policy Interventions (Draw Here)", type="pil", tool="sketch")
                
                # Drawing tools
                with gr.Row():
                    policy_type = gr.Dropdown(
                        choices=list(POLICY_TYPES.keys()),
                        value="Bike Lane",
                        label="Intervention Type"
                    )
                    line_width = gr.Slider(10, 100, value=40, label="Line Width (for lanes)")
                
                # Controls
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous Frame")
                    next_btn = gr.Button("➡️ Next Frame")
                    clear_btn = gr.Button("🗑️ Clear Policy Map")
                    save_btn = gr.Button("💾 Save Policy Map", variant="primary")
                
                frame_counter = gr.Textbox(label="Frame", value="1 / ?")
            
            with gr.Column(scale=1):
                gr.Markdown("### Instructions")
                gr.Markdown("""
                1. Select intervention type from dropdown
                2. Draw on the policy canvas:
                   - **Lines** for bike/bus lanes
                   - **Polygons** for zones
                   - **Circles** for charging stations
                3. Adjust line width if needed
                4. Click "Save" when done
                5. Move to next frame
                
                **Tips:**
                - Draw directly on the black policy map
                - Colors indicate intervention type
                - Saved maps will be used for generation
                """)
                
                status_box = gr.Textbox(label="Status", value="Ready")
        
        # Event handlers
        def load_frame():
            frame = editor.get_current_frame()
            policy_map = editor.create_blank_policy_map(frame.size)
            counter = f"{editor.current_frame_idx + 1} / {len(editor.frame_files)}"
            return frame, policy_map, counter
        
        def next_frame():
            editor.current_frame_idx = min(editor.current_frame_idx + 1, len(editor.frame_files) - 1)
            return load_frame()
        
        def prev_frame():
            editor.current_frame_idx = max(editor.current_frame_idx - 1, 0)
            return load_frame()
        
        def clear_policy():
            frame = editor.get_current_frame()
            policy_map = editor.create_blank_policy_map(frame.size)
            return policy_map, "Policy map cleared"
        
        def save_policy(policy_img, intervention_type):
            if policy_img is None:
                return "⚠️ No policy map to save"
            
            metadata = {
                "frame": editor.frame_files[editor.current_frame_idx].name,
                "intervention_type": intervention_type,
                "timestamp": str(Path.ctime(editor.frame_files[editor.current_frame_idx]))
            }
            
            editor.save_policy_map(policy_img, metadata)
            return f"✅ Saved policy for {metadata['frame']}"
        
        # Connect events
        demo.load(load_frame, outputs=[frame_display, policy_canvas, frame_counter])
        prev_btn.click(prev_frame, outputs=[frame_display, policy_canvas, frame_counter])
        next_btn.click(next_frame, outputs=[frame_display, policy_canvas, frame_counter])
        clear_btn.click(clear_policy, outputs=[policy_canvas, status_box])
        save_btn.click(save_policy, inputs=[policy_canvas, policy_type], outputs=[status_box])
    
    return demo


def main():
    if not GRADIO_AVAILABLE:
        print("❌ Gradio is required. Install with: pip install gradio")
        return
    
    parser = argparse.ArgumentParser(description="Interactive policy editor")
    parser.add_argument("--frames", type=str, required=True, help="Frames directory")
    parser.add_argument("--out", type=str, required=True, help="Output directory for policy maps")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    args = parser.parse_args()
    
    # Create editor
    editor = PolicyEditor(args.frames, args.out)
    
    # Launch Gradio interface
    demo = create_gradio_interface(editor)
    
    print(f"\n🚀 Launching policy editor...")
    print(f"   Open http://localhost:{args.port} in your browser")
    print(f"   Press Ctrl+C to stop\n")
    
    demo.launch(server_port=args.port, share=False)


if __name__ == "__main__":
    main()
