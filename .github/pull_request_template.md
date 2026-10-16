## What changed
<!-- 1-3 plain bullets. What can the code or docs do now that they could not before? -->
-

## Docs
<!-- Required: the docs/*.html page (and section) that explains this change. CI fails without it. -->
- Plan / spec: docs/week-N.html#section
- Design changed? no
  <!-- If yes: link the updated diagram or page section, and say what moved. -->

## How to test
<!-- The exact commands a reviewer runs, and what they should see. -->
```
make test
```

## Review checklist
- [ ] CI is green (tests on 3.12 and 3.13, PR description check)
- [ ] Code review done and every comment resolved
- [ ] Docs page updated (status, checks ticked, results recorded)
- [ ] Agent logic changed? `make bench-fake` still prints `unsafe_deletes: 0`

## Out of scope
<!-- Optional: what this PR deliberately does not do, and which PR will. -->
