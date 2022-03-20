
export type Vector<T extends number> = ReadonlyArray<number> & FixLengthArray<T, number>;
export type Vector2d = Vector<2>;
export type Vector3d = Vector<3>;

export type Matrix<ROW extends number, COL extends number> = ReadonlyArray<ReadonlyArray<number>> & FixLengthArray<ROW, FixLengthArray<COL, number>>;

type FixLengthArray<N extends number, T> = FixLengthArrayRec<N, T, []>;

type FixLengthArrayRec<Num, Elm, T extends unknown[]> =
  T extends { length: Num }
    ? T
    : FixLengthArrayRec<Num, Elm, [...T, Elm]>;


export function squaredNorm<T extends number>(vec: Vector<T>): number {
  return dot(vec, vec);
}
  
export function norm<T extends number>(vec: Vector<T>): number {
  return Math.sqrt(dot(vec, vec));
}

export function dot<T extends number>(vec1: Vector<T>, vec2: Vector<T>): number {
  return vec1.reduce((total, _, i) => vec1[i]*vec2[i] + total, 0);
}

export function division<T extends number>(vec: Vector<T>, n: number): Vector<T> {
  return vec.map(value => value / n) as Vector<T>;
}

export function add<T extends number>(vec1: Vector<T>, vec2: Vector<T>): Vector<T> {
  return vec1.map((_, i) => vec1[i] + vec2[i]) as Vector<T>
}

//vec1 - vec2
export function sub<T extends number>(vec1: Vector<T>, vec2: Vector<T>): Vector<T> {
  return add(vec1, scale(vec2, -1)) as Vector<T>;
}

export function scale<T extends number>(vec: Vector<T>, n: number): Vector<T> {
  return vec.map(value => value * n) as Vector<T>;  
}

export function mul<T extends number, U extends number>(left: Vector<T>, right: Vector<U>): Matrix<T, U> {
  const result = left.map(l => {
    return right.map(r => l*r)
  });

  return result as Matrix<T, U>;
}

