Reviewer: Astra. Stage: resource-only preflight before fitting. Inspected profile_v1.json (stable source maps retained in that profile).

Original reviewer-facing report:

No material objection. A **fixed 24-hour deadline with a properly warmed estimate** is reasonable for the complete 13-fit study; preserving three seeds is preferable to truncating the design.

Before fitting:

- Freeze the revised profiling method and deadline. Count startup once per fit, not once per optimizer step.
- Include all validation passes, checkpoint replay, data loading, checkpoint writes and collection.
- Keep at least 2× headroom over the measured total estimate. Two timed steps give a preliminary estimate, not a guaranteed overnight duration.
- Retain the 6 GiB/no-swap/two-thread limits and GPU cap. A failed or timed-out grid remains incomplete; do not reinterpret it as a smaller completed experiment.

This resource-only change occurs before outcomes and does not compromise preregistration. No additional user approval is scientifically necessary within the authorized local-work scope.
