export function generateSerialNumbers(length: number): number[] {
  return [...Array(length)].map((_, j) => j);
}
