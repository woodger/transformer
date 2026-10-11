import { createHash } from "node:crypto";

export function canonicalize(value) {
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
      .map((key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`);
    return `{${properties.join(",")}}`;
  }
  throw new TypeError(`Unsupported JCS value: ${typeof value}`);
}

export function sha256(value) {
  return createHash("sha256").update(canonicalize(value), "utf8").digest("hex");
}
