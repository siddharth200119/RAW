# Decision models (Jev and others)

A decider answers typed questions about a `state` with probabilities, never text. `OpenRouterDecider` calls OpenRouter's Decisions API (default model `typesafe/jev-1.14`; any decision model on OpenRouter works via `model=`). It needs only the base install.

```python
from RAW.deciders import OpenRouterDecider
from RAW.models import NoulQuestion, NoulCriteria, ChoiceQuestion, ScoreQuestion

decider = OpenRouterDecider()          # model="typesafe/jev-1.14", api_key from OPENROUTER_API_KEY
decision = await decider.decide(
    state={"ticket": "Checkout shows a blank screen after I click Pay."},
    questions={
        "is_bug": NoulQuestion(
            instructions="Is the customer reporting a software defect?",
            criteria=NoulCriteria(true="Broken product behaviour.", false="A question or feature request."),
        ),
        "team": ChoiceQuestion(
            instructions="Which team should own this ticket?",
            criteria={"payments": "Checkout or billing issues.", "frontend": "Rendering issues."},
        ),
        "urgency": ScoreQuestion(
            instructions="How urgent is this ticket?",
            criteria=["Can wait", "This week", "Blocking revenue now"],
        ),
    },
)
decision.answers["is_bug"].noul        # probability of yes
decision.answers["team"].choice        # winning key, plus .confidence and .probabilities
decision.answers["urgency"].score      # position on the scale
decision.usage.cost                    # USD
await decider.stop()
```

| Question | Answer field |
|---|---|
| `NoulQuestion(instructions, criteria=NoulCriteria(true, false))` | `.noul` probability of yes |
| `ChoiceQuestion(instructions, criteria={key: description})` | `.choice`, `.confidence`, `.probabilities` |
| `ScoreQuestion(instructions, criteria=[low ... high])` | `.score` |
| `BoxQuestion` (Jev 1.14+, images) | bounding box; it never reports "not found", so pair it with a `NoulQuestion` |

Rules: pick thresholds from labelled data and re-tune them per model; deciders return typed answers and no text; images in `state` need the `[agentic]` extra.

## Routing with a decider

```python
from RAW.utils import Router
router = Router(decider=OpenRouterDecider(), decider_threshold=0.5)
agent = Agent(name="support", llm=llm, tools=tools, skills=skills, router=router)
```

Each tool and skill gets a yes/no question and is selected when its probability is at least `decider_threshold`. The routing result also has per-tool and per-skill `probabilities` for tuning. Deciders produce no text, so `conversation_summary` is passed through unchanged.

## Testing

Subclass `BaseDecider` (`from RAW.deciders import BaseDecider`), implement `async decide(state, questions) -> Decision` and `async stop()`, and return canned `Decision`s.
