# Prompt Injection Detection Indicators

The `snagline.prompt_injection` module provides lightweight, deterministic heuristics for detecting common prompt injection and adversarial directive patterns in untrusted content before execution.

## Supported Indicator Rules

1. **`rule_instruction_override`**:
   Flags directives attempting to cancel, override, or disregard system instructions (e.g., *"ignore all previous instructions"*, *"new instructions follow"*).
2. **`rule_jailbreak_roleplay`**:
   Flags adversarial jailbreak personas and restriction-stripping roleplays (e.g., *"DAN mode"*, *"developer mode enabled"*).
3. **`rule_secret_exfiltration`**:
   Flags explicit attempts to compel the model to leak its system prompt, API keys, tokens, or credentials.
4. **`rule_tool_manipulation`**:
   Flags commands that attempt to manipulate tools into running silently or bypassing approval gates.

## Configuration & Policy Gate Integration

To integrate prompt injection checks into your pre-execution authorization policy:

```python
from snagline.authorize import Policy
from snagline.prompt_injection import as_authorize_rule, PromptInjectionConfig

config = PromptInjectionConfig(
    min_severity="medium",
    disabled_rules={"rule_jailbreak_roleplay"},
    trusted_keys={"system_prompt", "trusted_context"},
)

policy = Policy(
    rules=(as_authorize_rule(config=config, on_match="deny"),),
    default_decision="allow",
)
```

## Security Limitations & Non-Claims

> [!WARNING]
> Heuristic pattern matching is **advisory** and cannot reliably detect all indirect or obfuscated prompt injections. A clean verdict is never proof that an input is benign. Prompt injection indicators should serve as defense-in-depth alongside strict tool allowlists, least-privilege scoping, and approval barriers.
