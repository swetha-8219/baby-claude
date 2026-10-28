# Baby Claude — test report

Run: 2026-10-27, CPU only, in this VM.

Buffett checkpoint: dataset=buffett-clean, params=817,152, vocab=103, checkpoint final_loss=1.0989
Fresh loss check (20 new batches each, seed 999): train 1.0867, val 1.3781
The fresh check should land close to the checkpoint final_loss above, which confirms the saved checkpoint is the trained model, not a random one.

## Buffett generation tests (150 new tokens each)
### Prompt 1, temperature 0.7: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

• We will have this operate to the annual meeting at NFM, and we will be spent for more than 50 years on September 31,
the transaction of new investor

### Prompt 1, temperature 1.0: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

When Neval you watcomy now from the audience of these programs, are now to play wo sutce at a lose operation. February, with 
us only require through 

### Prompt 2, temperature 0.7: 'A wonderful business at a fair price is '

that the
doctor who will be in a rought annual earnings that will be attlent year. By she having a local cost of the 
book beyond that will allowad th

### Prompt 2, temperature 1.0: 'A wonderful business at a fair price is '

on their 
financing last your cash. The directors have importanced outstanding chemical system as supplement.
At Berkshire, that’s unleading will surp

### Prompt 3, temperature 0.7: 'Price is what you pay. Value is what you get. '


 
 
 Focus on September on Dave 
 
 
 
 Our test o of our trade best that will be about 4% on Ariel Recement (and First 
Compensation for Berkshire S

### Prompt 3, temperature 1.0: 'Price is what you pay. Value is what you get. '


 
What has smaller burdged on MidAmerican Energy, the Boardrar was looking float and them will choose the
corporate off. That operates our contrling,

### Prompt 4, temperature 0.7: 'Berkshire'

’s operations have delivered such as a wonderful purchase and
industry that the investment taxes of $1.7 billion on posses of $109 million and since t

### Prompt 4, temperature 1.0: 'Berkshire'

’s subset side
in most spent as often many substantial management. Consota3
rates, I bought far more than in results t han attributable t o this g pic

## Shakespeare checkpoint, for comparison
Checkpoint final_loss=2.3075. Prompt 'ROMEO:', temperature 0.8:

ROMEO:
And my thicee, thand ase orain: melyss thee ond ais beeal
The ourou sture, on theast then ballle ofes we and ithethis t and fat.

Hor blindor thit mp

## Edge case: character outside the vocab
Prompt 'Berkshire 😀' (emoji): missing chars = ['😀']. The sampler reports them instead of silently mapping them to a wrong character.

Total test time: 22s