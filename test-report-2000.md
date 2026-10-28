# Baby Claude — test report

Run: 2026-10-27, CPU only, in this VM.

Buffett checkpoint: dataset=buffett, params=817,280, vocab=104, checkpoint final_loss=1.3173
Fresh loss check (20 new batches each, seed 999): train 1.3410, val 1.3358
This matches the end-of-training eval (train 1.3456 / val 1.3503) closely, so the saved checkpoint is the trained model, not a random one.

## Buffett generation tests (150 new tokens each)
### Prompt 1, temperature 0.7: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

But In all one the stures of companies the reason the about of the matters. But the and bighamen the many income a ound
thereas we much the operating 

### Prompt 1, temperature 1.0: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

Internations
Charlie and I stand, Berkshire smally is are a believe itiler of an she made time, handing we have briensk. As there, a
can cost of of yo

### Prompt 2, temperature 0.7: 'A wonderful business at a fair price is '

that our betwing shareholders.
A bought help as the commition we have been so currently in these corectaters whose and the come to ough the come of re

### Prompt 2, temperature 1.0: 'A wonderful business at a fair price is '

insurance rate businesses were
facused by the most changer, however, about been the ellyicative exchicelars cost and other throughly.
Berkshire’s brin

### Prompt 3, temperature 0.7: 'Price is what you pay. Value is what you get. '

We will tree for the can enconomy, which faming the results and the early and
the businessees of our at that wish posibilities appricable, tent will b

### Prompt 3, temperature 1.0: 'Price is what you pay. Value is what you get. '

Fe’s buying, that will give, my with benefit the can’t be in service of
the sayes indoed by to the case throwing coours. First, act teax-timen spurlas

### Prompt 4, temperature 0.7: 'Berkshire'

’s service the purchased a real find the by bade far our business and for the remarketable as the
insurance prospers and the market companies of our t

### Prompt 4, temperature 1.0: 'Berkshire'

 tax calcuments. The new
calls out charged by the face transagins float world have hand all a good on at outlike famined. Try the aggativates and
you 

## Shakespeare checkpoint, for comparison
Checkpoint final_loss=2.3075. Prompt 'ROMEO:', temperature 0.8:

ROMEO:
And my thicee, thand ase orain: melyss thee ond ais beeal
The ourou sture, on theast then ballle ofes we and ithethis t and fat.

Hor blindor thit mp

## Edge case: character outside the vocab
Prompt 'Berkshire 😀' (emoji): missing chars = ['😀']. The sampler reports them instead of silently mapping them to a wrong character.

Total test time: 20s