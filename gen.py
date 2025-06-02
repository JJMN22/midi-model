import torch
import numpy as np
import MIDI
from midi_model import MIDIModel, MIDIModelConfig

def load_model(checkpoint_path, config_name="tv2o-medium"):
    """Load your trained model"""
    config = MIDIModelConfig.from_name(config_name)
    model = MIDIModel(config=config)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    return model

def generate_midi_from_scratch(model, output_path="generated.mid", num_files=5):
    """Generate MIDI files from scratch"""
    
    print(f"Generating {num_files} MIDI files...")
    
    for i in range(num_files):
        # Generate from scratch (no prompt)
        midi_tokens = model.generate(
            batch_size=1,           # Generate 1 file at a time
            max_len=512,           # Length (adjust as needed)
            temp=1.0,              # Temperature (creativity)
            top_p=0.98,            # Nucleus sampling
            top_k=20               # Top-k sampling
        )
        
        # Convert tokens back to MIDI
        midi_score = model.tokenizer.detokenize(midi_tokens[0])
        
        # Save as MIDI file
        filename = f"{output_path.replace('.mid', '')}_{i+1}.mid"
        with open(filename, 'wb') as f:
            f.write(MIDI.score2midi(midi_score))
        
        print(f"Generated: {filename}")

if __name__ == "__main__":
    # Load your trained model
    model = load_model("lightning_logs/version_0/checkpoints/last.ckpt")
    
    # Generate MIDI files from scratch
    generate_midi_from_scratch(model, "my_generated_music.mid", num_files=3)
