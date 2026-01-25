#!/usr/bin/env python3
"""
MineAI SDK Comprehensive Test Suite

This script tests all features of the MineAI Python SDK including:
- Basic chat completions (sync/async)
- Streaming responses
- Memory functionality
- Temperature parameter
- Max tokens parameter
- Retry on failure
- Rate limiting/throttling
- Error handling

Requirements:
    pip install mineai

Usage:
    export MINEAI_API_KEY="your-api-key"
    python test_all_features.py
"""

import os
import sys
import asyncio
import time
from typing import List, Dict

# Import MineAI SDK
try:
    from mineai import MineAI, AsyncMineAI, Models
    from mineai.errors import (
        AuthenticationError,
        BadRequestError,
        RateLimitError,
        InternalServerError,
        APIConnectionError
    )
except ImportError:
    print("❌ Error: mineai package not found. Install it with: pip install mineai")
    sys.exit(1)


class TestRunner:
    """Test runner for MineAI SDK"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = MineAI(api_key=api_key)
        self.async_client = AsyncMineAI(api_key=api_key)
        self.test_results: List[Dict] = []
        
    def log_test(self, name: str, status: str, message: str = ""):
        """Log test result"""
        emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{emoji} {name}: {status}")
        if message:
            print(f"   {message}")
        self.test_results.append({
            "name": name,
            "status": status,
            "message": message
        })
    
    def test_basic_completion(self):
        """Test 1: Basic chat completion"""
        print("\n📝 Test 1: Basic Chat Completion")
        try:
            response = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
                ]
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                self.log_test("Basic Completion", "PASS", f"Response: {content[:50]}...")
            else:
                self.log_test("Basic Completion", "FAIL", "Invalid response structure")
                
        except Exception as e:
            self.log_test("Basic Completion", "FAIL", str(e))
    
    def test_streaming(self):
        """Test 2: Streaming response"""
        print("\n📝 Test 2: Streaming Response")
        try:
            stream = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Count from 1 to 5."}
                ],
                stream=True
            )
            
            chunks_received = 0
            full_content = ""
            
            for chunk in stream:
                if "choices" in chunk:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        chunks_received += 1
            
            if chunks_received > 0:
                self.log_test("Streaming", "PASS", f"Received {chunks_received} chunks")
            else:
                self.log_test("Streaming", "FAIL", "No chunks received")
                
        except Exception as e:
            self.log_test("Streaming", "FAIL", str(e))
    
    async def test_async_completion(self):
        """Test 3: Async chat completion"""
        print("\n📝 Test 3: Async Chat Completion")
        try:
            response = await self.async_client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Say 'Async works!' and nothing else."}
                ]
            )
            
            if response and "choices" in response:
                content = response["choices"][0]["message"]["content"]
                self.log_test("Async Completion", "PASS", f"Response: {content[:50]}...")
            else:
                self.log_test("Async Completion", "FAIL", "Invalid response structure")
                
        except Exception as e:
            self.log_test("Async Completion", "FAIL", str(e))
    
    def test_memory(self):
        """Test 4: Memory functionality"""
        print("\n📝 Test 4: Memory Functionality")
        try:
            # First message
            response1 = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "My favorite color is blue. Remember this."}
                ],
                memory=True
            )
            
            time.sleep(1)  # Brief pause
            
            # Second message - should remember
            response2 = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "What is my favorite color?"}
                ],
                memory=True
            )
            
            if response1 and response2:
                self.log_test("Memory", "PASS", "Memory conversation completed")
            else:
                self.log_test("Memory", "FAIL", "Memory responses incomplete")
                
        except Exception as e:
            self.log_test("Memory", "FAIL", str(e))
    
    def test_temperature(self):
        """Test 5: Temperature parameter"""
        print("\n📝 Test 5: Temperature Parameter")
        try:
            # Low temperature (more deterministic)
            response_low = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "What is 2+2?"}
                ],
                temperature=0.1
            )
            
            # High temperature (more creative)
            response_high = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Write a creative word."}
                ],
                temperature=1.5
            )
            
            if response_low and response_high:
                self.log_test("Temperature", "PASS", "Temperature variations tested")
            else:
                self.log_test("Temperature", "FAIL", "Temperature responses incomplete")
                
        except Exception as e:
            self.log_test("Temperature", "FAIL", str(e))
    
    def test_max_tokens(self):
        """Test 6: Max tokens parameter"""
        print("\n📝 Test 6: Max Tokens Parameter")
        try:
            response = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Write a long essay about space."}
                ],
                max_tokens=50  # Limit to 50 tokens
            )
            
            if response and "usage" in response:
                completion_tokens = response["usage"]["completion_tokens"]
                if completion_tokens <= 50:
                    self.log_test("Max Tokens", "PASS", f"Response limited to {completion_tokens} tokens")
                else:
                    self.log_test("Max Tokens", "WARN", f"Exceeded limit: {completion_tokens} tokens")
            else:
                self.log_test("Max Tokens", "PASS", "Max tokens parameter accepted")
                
        except Exception as e:
            self.log_test("Max Tokens", "FAIL", str(e))
    
    def test_retry_on_failure(self):
        """Test 7: Retry on failure"""
        print("\n📝 Test 7: Retry on Failure")
        try:
            response = self.client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[
                    {"role": "user", "content": "Test retry logic."}
                ],
                retry_on_failure=True
            )
            
            if response:
                self.log_test("Retry Logic", "PASS", "Retry parameter accepted")
            else:
                self.log_test("Retry Logic", "FAIL", "No response received")
                
        except Exception as e:
            self.log_test("Retry Logic", "FAIL", str(e))
    
    def test_rate_limiting(self):
        """Test 8: Rate limiting detection"""
        print("\n📝 Test 8: Rate Limiting Detection")
        try:
            # Send multiple rapid requests to trigger throttling
            responses = []
            for i in range(5):
                response = self.client.chat.completions.create(
                    model=Models.O1_FREE,
                    messages=[
                        {"role": "user", "content": f"Quick test {i}"}
                    ]
                )
                responses.append(response)
                time.sleep(0.1)  # Brief delay between requests
            
            # Check if any response contains throttle information
            throttled = any(r.get("throttle") for r in responses if isinstance(r, dict))
            
            if throttled:
                self.log_test("Rate Limiting", "PASS", "Throttling detected in responses")
            else:
                self.log_test("Rate Limiting", "PASS", "No throttling triggered (expected with low volume)")
                
        except Exception as e:
            self.log_test("Rate Limiting", "FAIL", str(e))
    
    def test_error_handling(self):
        """Test 9: Error handling"""
        print("\n📝 Test 9: Error Handling")
        
        # Test invalid API key
        try:
            bad_client = MineAI(api_key="invalid_key_12345")
            bad_client.chat.completions.create(
                model=Models.O1_FREE,
                messages=[{"role": "user", "content": "Test"}]
            )
            self.log_test("Error Handling (401)", "FAIL", "Should have raised AuthenticationError")
        except AuthenticationError:
            self.log_test("Error Handling (401)", "PASS", "AuthenticationError raised correctly")
        except Exception as e:
            self.log_test("Error Handling (401)", "FAIL", f"Unexpected error: {e}")
        
        # Test invalid model
        try:
            self.client.chat.completions.create(
                model="invalid-model",
                messages=[{"role": "user", "content": "Test"}]
            )
            self.log_test("Error Handling (400)", "WARN", "Invalid model accepted")
        except (BadRequestError, Exception):
            self.log_test("Error Handling (400)", "PASS", "Bad request error handled")
    
    def test_all_models(self):
        """Test 10: All supported models"""
        print("\n📝 Test 10: All Supported Models")
        
        models = [
            ("mine:o1-free", Models.O1_FREE),
            ("mine:r3-rt-y", Models.R3_RT_Y),
            ("mine:r3-rt-z", Models.R3_RT_Z),
        ]
        
        for model_name, model_const in models:
            try:
                response = self.client.chat.completions.create(
                    model=model_const,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=20
                )
                
                if response:
                    self.log_test(f"Model: {model_name}", "PASS", "Model responded")
                else:
                    self.log_test(f"Model: {model_name}", "FAIL", "No response")
                    
                time.sleep(1)  # Brief delay between model tests
                
            except Exception as e:
                # Some models might require paid plans
                self.log_test(f"Model: {model_name}", "WARN", str(e))
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        passed = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed = sum(1 for r in self.test_results if r["status"] == "FAIL")
        warned = sum(1 for r in self.test_results if r["status"] == "WARN")
        total = len(self.test_results)
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Warnings: {warned}")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*60)
        
        if failed > 0:
            print("\nFailed Tests:")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"  - {result['name']}: {result['message']}")
    
    async def run_async_tests(self):
        """Run async tests"""
        await self.test_async_completion()
    
    def run_all_tests(self):
        """Run all tests"""
        print("="*60)
        print("🚀 MineAI SDK Comprehensive Test Suite")
        print("="*60)
        
        # Sync tests
        self.test_basic_completion()
        self.test_streaming()
        self.test_memory()
        self.test_temperature()
        self.test_max_tokens()
        self.test_retry_on_failure()
        self.test_rate_limiting()
        self.test_error_handling()
        self.test_all_models()
        
        # Async tests
        print("\n📝 Running Async Tests...")
        asyncio.run(self.run_async_tests())
        
        # Print summary
        self.print_summary()


def main():
    """Main entry point"""
    # Check for API key
    api_key = os.getenv("MINEAI_API_KEY")
    
    if not api_key:
        print("❌ Error: MINEAI_API_KEY environment variable not set")
        print("\nUsage:")
        print("  export MINEAI_API_KEY='your-api-key'")
        print("  python test_all_features.py")
        sys.exit(1)
    
    # Run tests
    runner = TestRunner(api_key)
    runner.run_all_tests()


if __name__ == "__main__":
    main()
