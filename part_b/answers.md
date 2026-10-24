# Part B

## Question 1

PR 1: the block is right. The cited tests check FX locking and four-decimal rendering, not whether a cached display is reused or invalidated. The tool blocked only because the string "cache" was absent. A correct check must read the caching change in the diff and an assertion about a hit, expiry, or refresh.

PR 2: the pass is wrong. Rollback, callback, timeout, and funding source appear in separate titles. A configured 5000ms value can pass while a late callback still settles, and "handles provider callback" does not say the callback failed. A correct check needs one behavior: after five seconds without the callback, the credit is reversed and the funding source is not debited.

PR 3: the block is right for this change, but the reason is weak. "Large top-ups" and "top-up succeeds" never assert the corrected boundary. The tool blocked because "threshold" and "SGD" were absent, so it would also block a precise test that says "approval is required above 10000." A correct check compares the old and new threshold with cases on both sides of it.

## Question 2

I would not ask the team to trust a raw model verdict. The model may label requirements and quote evidence; Python applies the rules for amounts, the five-second timeout, and the approver, then calculates the decision. A second model may downgrade labels but cannot raise them. CI pins the model digest and prompt hash and uses temperature 0. I would measure repeated-run range, verdict-flip rate, requirement agreement, JSON failures, and disagreement with reviewed pull requests. On the Part A evaluator, five repeats scored 37.5–56.25, the standard deviation was 8.15, and the block at 75 never flipped. If a future suite straddles the threshold, I move the threshold below that noise band or require two agreeing runs. A missing critical requirement still blocks whatever the score is.

## Question 3

I would retrieve the uncovered requirement IDs, the spec sentences behind them, the pull-request diff, the changed functions and their callers, the nearest existing tests, and one or two style examples. The prompt would name the test framework, forbid production edits, and require every test to name its requirement, arrange state, act once, and assert an observable result, including boundary values. I would run the generated tests, revert the fix and confirm that the new test fails, then score its text with this evaluator. A test that passes on the unpatched code, or merely repeats the spec, is rejected. I expect failures around timing, invented helpers, fixtures that do not compile, and assertions that preserve the implementation's current mistake. Generation can propose a patch; it cannot merge itself.
