// Typed access to the curated quality dataset.
//
// quality-data.json is generated from reports/*.json by scripts/build-quality-data.mjs
// (run via the prebuild/predev npm hooks). It is imported at build time, so the
// quality page needs no network request and cannot fail to load.

import data from "./quality-data.json";
import type { QualityData } from "./types";

export const quality = data as QualityData;
