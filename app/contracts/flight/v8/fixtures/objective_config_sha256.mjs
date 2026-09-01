#!/usr/bin/env node

import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";

function canonicalize(value) {
  if (value === null || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new TypeError("JCS numbers must be finite");
    }
    return JSON.stringify(value);
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(",")}]`;
  }
  if (typeof value === "object") {
    const properties = Object.keys(value)
      .sort()
      .map(
        (key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`,
      );
    return `{${properties.join(",")}}`;
  }
  throw new TypeError(`Unsupported JCS value: ${typeof value}`);
}

const [fixturePath] = process.argv.slice(2);
if (fixturePath === undefined) {
  throw new TypeError("usage: objective_config_sha256.mjs <fixture.json>");
}

const objective = JSON.parse(await readFile(fixturePath, "utf8"));
const document = {
  targets: [
    "MeanReturn",
    "SigmaReturn",
    "ProbTP",
    "ProbSL",
    "VolatilityNext",
    "HittingProbTP",
  ],
  objective,
};
const canonical = canonicalize(document);
const digest = createHash("sha256").update(canonical, "utf8").digest("hex");
process.stdout.write(`${digest}\n`);
