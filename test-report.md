# Baby Claude — test report

Run: 2026-10-27, CPU only, in this VM.

Buffett checkpoint: dataset=buffett, params=817,280, vocab=104, checkpoint final_loss=2.4058
Fresh loss check (20 new batches each, seed 999): train 2.3476, val 2.2132
This matches the end-of-training eval (train 2.3438 / val 2.2748) closely, so the saved checkpoint is the trained model, not a random one.

## Buffett generation tests (150 new tokens each)
### Prompt 1, temperature 0.7: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

Berkearshatorers t tourthe as has t thit theres wiresinat blourstind te ther inthe tantinind we the s ans incore fie fowis s t windind othen arshenchi

### Prompt 1, temperature 1.0: 'To the Shareholders of Berkshire Hathaway Inc.:\n\n'

Intesirer tshis te wh fars ss thay tarestime homerave arsimpangreay itiverto the sse oway tindeses suld we ithe breengero ste tr fis
cthouns, oriofry 

### Prompt 2, temperature 0.7: 'A wonderful business at a fair price is '

thanond to blilinesen aned arst benourse st can thesto wine inses we indengst cks st owillurker othie thas a th weasins arecore s fowis insen be pore 

### Prompt 2, temperature 1.0: 'A wonderful business at a fair price is '

ino oncie f onol astititoulenasom
iof ar t the of iche witin th tanalll wicowin tintilly comoldinc s arkare cean antrerind t tunoneand t ay, alees st 

### Prompt 3, temperature 0.7: 'Price is what you pay. Value is what you get. '

withir thas bl fere blange co comen mave mas sisusotr tis bondind se thenden o cl busthe the coman seathe an by t fos our tiveand athasese seton bllis

### Prompt 3, temperature 1.0: 'Price is what you pay. Value is what you get. '

ond se timan teartsin wime a mes tof berere athe wathe fince seran cur avely atecer bosea imonerert bell traliotoneluss. ings tenco ceasting ting s ai

### Prompt 4, temperature 0.7: 'Berkshire'

ed thaly ale owe of theand is thit te bompan ane faned inthe thetome ind ath orke s thaleanco is inos t ind th sun carissthing tred con we or cinomome

### Prompt 4, temperature 1.0: 'Berkshire'

ntary bupe thathiome t secarore sied indede f tinclas cheang pres flests. a o m athen be bura ind sof wareas fin fanintit ithathedely coucans anes it 

## Shakespeare checkpoint, for comparison
Checkpoint final_loss=2.3075. Prompt 'ROMEO:', temperature 0.8:

ROMEO:
And my thicee, thand ase orain: melyss thee ond ais beeal
The ourou sture, on theast then ballle ofes we and ithethis t and fat.

Hor blindor thit mp

## Edge case: character outside the vocab
Prompt 'Berkshire 😀' (emoji): missing chars = ['😀']. The sampler reports them instead of silently mapping them to a wrong character.

Total test time: 19s