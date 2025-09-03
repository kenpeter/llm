#!/usr/bin/env python3

import chainlit as cl
import torch
import tiktoken
import os
import asyncio
from pretrain_flash2 import GPTModel, GPT_CONFIG

class CustomGPTInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, checkpoint_path="checkpoints/latest_checkpoint.pt"):
        """Load custom trained GPT-2 weights"""
        print("Loading custom trained GPT-2 model...")
        
        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            available_checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
            if available_checkpoints:
                checkpoint_path = f"checkpoints/{available_checkpoints[0]}"
                print(f"Using available checkpoint: {checkpoint_path}")
            else:
                raise FileNotFoundError("No checkpoint files found in checkpoints/ directory")
        
        # Load trained weights and config
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Use the config from checkpoint if available
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            config = GPT_CONFIG
            
        # Load model architecture with correct config
        self.model = GPTModel(config)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path} on {self.device}")
        
    def generate_text(self, prompt, max_tokens=100, temperature=0.8, top_k=50):
        """Generate text using the custom trained model"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please wait for initialization."
            
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        
        generated_ids = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # Get model predictions  
                context_length = getattr(self.model, 'context_length', 512)
                if len(generated_ids) > context_length:
                    # Truncate to context length
                    context_ids = generated_ids[-context_length:]
                else:
                    context_ids = generated_ids
                    
                context_tensor = torch.tensor([context_ids], device=self.device)
                
                with torch.no_grad():
                    logits = self.model(context_tensor)
                    logits = logits[0, -1, :]  # Get logits for last token
                    
                    # Apply temperature
                    logits = logits / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, top_k)
                        logits_filtered = torch.full_like(logits, float('-inf'))
                        logits_filtered[top_k_indices] = top_k_logits
                        logits = logits_filtered
                    
                    # Sample next token
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    # Stop if we hit end of text token
                    if next_token == self.tokenizer.eot_token:
                        break
                        
                    generated_ids.append(int(next_token))
        
        # Decode only the generated portion
        generated_tokens = generated_ids[len(input_ids):]
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return generated_text

# Initialize inference engine
gpt_inference = CustomGPTInference()

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(
        content="🚀 **Custom Trained GPT-2**\n\nInitializing GPT-2 with your custom trained weights...",
    ).send()
    
    # Load model in background
    try:
        await asyncio.get_event_loop().run_in_executor(None, gpt_inference.load_model)
        await cl.Message(
            content="✅ **Ready!**\n\nCustom GPT-2 model loaded successfully. You can now start chatting with your trained model!",
        ).send()
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error loading model:** {str(e)}\n\nMake sure you have trained the model by running the chapter 5 notebook first.",
        ).send()

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_prompt = message.content
    
    # Show thinking message
    thinking_msg = cl.Message(content="🤔 Generating response with your custom model...")
    await thinking_msg.send()
    
    try:
        # Generate response
        response = await asyncio.get_event_loop().run_in_executor(
            None, 
            lambda: gpt_inference.generate_text(user_prompt, max_tokens=100, temperature=0.8)
        )
        
        # Update message with response
        thinking_msg.content = f"**Prompt:** {user_prompt}\n\n**Response:** {response}"
        await thinking_msg.update()
        
    except Exception as e:
        thinking_msg.content = f"❌ **Error generating response:** {str(e)}"
        await thinking_msg.update()

if __name__ == "__main__":
    import os
    os.system("chainlit run app_own.py --port 8000")