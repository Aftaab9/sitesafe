// Mirrors sitesafe/classes.py. The order is the model's output order and must not change.
export const CLASS_NAMES = [
  'helmet', 'gloves', 'vest', 'boots', 'goggles', 'Person',
  'no_helmet', 'no_goggle', 'no_gloves', 'no_boots', 'forklift', 'no_vest',
] as const;
export const NUM_CLASSES: number = CLASS_NAMES.length;
export const PERSON = 5;
export const FORKLIFT = 10;
/** A forklift box below this confidence is drawn but never raises a proximity alert or counts as a forklift. */
export const FORKLIFT_MIN_CONF = 0.5;
/** People are detected from this confidence (PPE items from the sensitivity setting, 0.25 by default). */
export const PERSON_MIN_CONF = 0.2;

export type Item = 'helmet' | 'goggles' | 'vest' | 'gloves' | 'boots';
export const PPE_ITEMS: Item[] = ['helmet', 'goggles', 'vest', 'gloves', 'boots'];
export const POS_CLASS: Record<Item, number> = { helmet: 0, goggles: 4, vest: 2, gloves: 1, boots: 3 };
export const NEG_CLASS: Record<Item, number> = { helmet: 6, goggles: 7, vest: 11, gloves: 8, boots: 9 };

export const ITEM_OF_CLASS: Record<number, { item: Item; worn: boolean }> = {};
for (const item of PPE_ITEMS) {
  ITEM_OF_CLASS[POS_CLASS[item]] = { item, worn: true };
  ITEM_OF_CLASS[NEG_CLASS[item]] = { item, worn: false };
}

export const ITEM_BAND: Record<Item, [number, number]> = {
  helmet: [0.0, 0.3],
  goggles: [0.0, 0.3],
  vest: [0.1, 0.75],
  gloves: [0.15, 0.9],
  boots: [0.75, 1.0],
};

export type ProfileId = 'general' | 'full' | 'cutting_grinding';
export const SITE_PROFILES: Record<ProfileId, { label: string; hint: string; required: Item[] }> = {
  general: { label: 'General site', hint: 'Helmet and hi-vis vest', required: ['helmet', 'vest'] },
  full: { label: 'Full PPE', hint: 'Helmet, vest, gloves, boots', required: ['helmet', 'vest', 'gloves', 'boots'] },
  cutting_grinding: {
    label: 'Cutting / grinding',
    hint: 'Full PPE plus goggles',
    required: ['helmet', 'vest', 'gloves', 'boots', 'goggles'],
  },
};

export const ITEM_LABEL: Record<Item, string> = {
  helmet: 'Helmet', goggles: 'Goggles', vest: 'Vest', gloves: 'Gloves', boots: 'Boots',
};
