# Design direction: "industrial HUD"

This is the direction the web app was built to. Change the tokens in `web/src/index.css` and `web/tailwind.config.js` to restyle it.

## Moodboard in words

A site control room at night. Matte charcoal panels, a faint blueprint grid on the wall, hazard tape along the top edge, and one bright colour that means "look here": high-visibility yellow. Camera viewfinder corner brackets frame every image. Numbers are set in a monospace face, like a gauge readout. Nothing glows for decoration; colour is spent only on status.

## Palette

| Role | Dark | Light |
|---|---|---|
| Background | `#0B0D10` | `#F4F3EE` |
| Surface / raised | `#13171C` / `#1C2229` | `#FFFFFF` / `#ECEAE2` |
| Line | `#2C343E` | `#D0CDC2` |
| Text / muted | `#E8ECEF` / `#8B98A5` | `#14181D` / `#5C636C` |
| Hi-vis (primary) | `#FFD400` | `#FFD400` |
| Hazard (zones, alerts) | `#FF7A1A` | `#D65C00` |
| Compliant | `#2FD67B` | `#0A8648` |
| PPE missing | `#FF5B5D` | `#C82026` |
| Not sure | `#5BC0FF` | `#0068B5` |

"Not sure" is blue on purpose: calm, and clearly separate from the yellow used for the brand and from red and green.

## Type

Space Grotesk for headings, Inter for interface text, JetBrains Mono for numbers, labels and eyebrows. All three are self-hosted, so the app works offline.

## Status is never colour alone

| Status | Colour | Box line | Symbol | Label |
|---|---|---|---|---|
| Compliant | green | solid | tick | Compliant |
| PPE missing | red | dashed | cross | PPE missing |
| Not sure | blue | dotted | question mark | Check manually |

## Three key screens

**Photo check.** Left: the image in viewfinder brackets. A yellow scan line sweeps down, then each worker's corner brackets snap in one after another with a short spring. Each box carries a tag (symbol, W1, missing item). Hovering a worker dims the others, draws dashed connector lines to their matched PPE boxes, and floats a safety card beside them: confidence ring, five PPE icons with tick, cross or question mark. Right: a half-circle compliance gauge, three counts, bars for missing items, then the site profile, the report buttons and settings. Under the image: privacy slider, draw danger zone, explain.

**Live camera.** Full screen. Corner brackets, a red "live, not recorded" dot, frames per second and latency in monospace at top left, the gauge at bottom left, round buttons for sound, camera flip and settings at top right, and one yellow Stop button.

**Clip check.** The video frame with overlays, then a timeline strip of one cell per analysed frame: green for clear, red hatched for a violation, orange for a zone alert. Clicking a cell jumps there. Below it, the anonymous worker log.

## Motion

Scan line 1.1 s; lock-on spring per worker with 120 ms stagger; gauge and bars ease over 0.5 to 0.7 s; danger zones pulse only when someone is inside. All of it stops when the system or the in-app "Reduce motion" setting is on.

## Tone of the words

Short, direct, written for the officer. Buttons say what they do ("Draw danger zone", "Lock in"). The not-sure state reads as a hand-over, never as an error: "could not be decided with confidence… please take a look yourself."
