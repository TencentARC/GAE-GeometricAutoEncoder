# Stage 2 modules

`GAEFlow` / `GAEFlowTemporal` are the public names.

Checkpoint tensors still follow the internal DiT-v4 layout, so the class
hierarchy is kept for `load_state_dict`:

```
GAEFlowTemporal  (dit_temporal.py)   frame-axis RoPE
  └── GAEFlow    (dit.py)            x-prediction + IG baseline head
        └── TokenConcatDDT            token-concat references
              └── DDTHead
                    └── DDT.py / lightningDiT.py   blocks, attention, embed
```

Do not instantiate `TokenConcatDDT` from configs; use
`stage2.models.dit_temporal.GAEFlowTemporal`.
