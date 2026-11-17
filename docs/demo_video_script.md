# Demo video script (3 to 5 minutes)

Record the screen at 1080p with the live app URL visible in the address bar. Speak in your own words; the lines below are prompts, not a text to read out. Three real cases, one of them a failure.

| Time | On screen | What to say |
|---|---|---|
| 0:00 to 0:20 | Landing page. Let the hero loop play once. | Who it is for: a site safety officer on a walk-round. What it decides: who is missing required PPE. Why privacy matters: the question is what is worn, not who wears it. |
| 0:20 to 0:35 | Turn on flight mode or open DevTools, Network tab. Click "Try a sample". | The model runs in the browser. Nothing is uploaded. Show that no request carries the image. |
| 0:35 to 1:20 | **Case 1, a mixed crew (sample "Mixed crew").** Scan line, boxes lock on. Hover one worker: connector lines and safety card. Drag the privacy slider. Switch the site profile to "Full PPE" and watch statuses change. | Green, red and blue boxes with line styles and symbols. How PPE is matched to a person. Faces blurred on the device. The required PPE depends on the site profile. |
| 1:20 to 1:50 | Same photo: "Draw danger zone" around the machine area, close the polygon. Then open the forklift sample. | Danger zones use the feet point. Forklift proximity is measured in person-heights because there is no depth. |
| 1:50 to 2:30 | **Case 2, a clip.** Clip tab, load a clip (your own real clip if you have one), press Analyse. Scrub the timeline, click a red marker. Show the worker log and the smoothing line. | Frame sampling, anonymous worker numbers that persist, and a violation only after 3 of 5 frames. Quote the processing time shown. |
| 2:30 to 3:10 | **Case 3, the failure (sample "Where it fails").** Read the orange failure note. Point at a worker the model got wrong or sent to "check manually". | Say plainly what went wrong and why (distance, overlap, low light). This is why the output is a prompt for the officer and why "not sure" exists. Quote the test numbers: share of violators shown green, share sent to not sure. |
| 3:10 to 3:40 | Weather Lab. Pick a photo, toggle between the two models. | The generative AI part: diffusion-made night, rain, fog and dust versions of real images, used for training. Give the measured change in recall on real test images with its interval, whatever it is. |
| 3:40 to 4:05 | Back on the photo page: "PDF report", open it. Then the Dashboard. | Aggregate counts, anonymous numbers, no names, nothing stored. |
| 4:05 to 4:30 | Phone in hand (or phone screen recording): Live tab, point at a colleague wearing a helmet, then without. | Works on a phone, shows fps and latency, not recorded. |
| 4:30 to 4:45 | Model card page, scroll to "Out of scope". | Not for monitoring or disciplining individuals. Supports the officer, does not replace them. |

## Before recording

- Open the app once so the model is cached, then reload.
- Use the deployed URL, not localhost.
- For the live shot, get consent from the person on camera.
- Keep the numbers you quote identical to the ones in the report and on the model card page.
