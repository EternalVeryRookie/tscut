import { Matrix, forEach } from "mathjs";
import { generateSerialNumbers } from "../libs/array";


type flowNetwork = {
  capacity: (i: number, j: number) => number
  vSize: number
  add: (path: number[], volume: number) => void //pathの容量をvolumeだけ減らす
}


export function mincut(adjacencyMatrix: Matrix, sourceIndex: number, terminalIndex: number): {
  source: number[],
  terminal: number[]
} {
  const f = makeFlow(adjacencyMatrix);

  while (true) {
    const path = depthSearch(f, sourceIndex, terminalIndex, [sourceIndex]);
    if (path === null) {
      break;
    }

    const minCapacity = minCap(f, path);
    f.add(path, minCapacity);
  }

  //sourceからたどれる頂点を列挙する。列挙されなかった頂点がterminal側
  const sources = generateSerialNumbers(adjacencyMatrix.size()[0]).reduce<number[]>((sources, to) => {
    if (to === sourceIndex) {
      return sources;
    }

    const path = depthSearch(f, sourceIndex, to, [sourceIndex]);
    if (path !== null) {
      sources.push(to);
    }

    return sources;
  }, [sourceIndex]);

  const terminals = generateSerialNumbers(adjacencyMatrix.size()[0]).filter(i => !sources.includes(i));

  return {
    source: sources,
    terminal: terminals
  }
}


//tIndexに到達可能ならそこまでの経路を、不可能ならnullをreturnする
function depthSearch(f: flowNetwork, from: number, tIndex: number, intermediatePath: number[]): number[] | null {
  //訪問済みの頂点は探索しない
  const toList = generateSerialNumbers(f.vSize).filter(i => f.capacity(from, i) > 0 && !intermediatePath.includes(i))

  return toList.reduce<number[] | null>((pre, to) => {
    if (pre !== null && pre[pre.length-1] === tIndex) {
      return pre;
    }
    const path = intermediatePath.concat([to]);
    if (to === tIndex) {
      return path;
    }

    return depthSearch(f, to, tIndex, path);
  }, null);
}


//pathで示される経路の内、最小容量を取得する。
function minCap(f: flowNetwork, path: number[] /*頂点のインデックスの配列で経路を表す*/): number {
  return path.reduce((min, from, index) => {
    if (index === path.length - 1) {
      return min;
    }

    const to = path[index+1];
    const cap = f.capacity(from, to);
    if (min > cap || min < 0) {
      return cap;
    }

    return min;
  }, -1);
}


//隣接行列からmin-cutアルゴリズム用のフローネットワークを作成する。
function makeFlow(AdjacencyMatrix: Matrix): flowNetwork {
  const AdjacencyMatrixCopy = AdjacencyMatrix.clone();

  forEach(AdjacencyMatrixCopy, (value: number, index: number[]) => {
    const diag = [index[1], index[0]];
    //頂点が双方向に連結しているグラフの場合、間に別の頂点を挟むことで双方向の連結をなくす
    if (value !== 0 && AdjacencyMatrixCopy.get(diag) !== 0) {
      const size = AdjacencyMatrixCopy.size();
      AdjacencyMatrixCopy.resize([size[0]+2, size[1] + 2]);

      AdjacencyMatrixCopy.set([size[0], index[1]], value);
      AdjacencyMatrixCopy.set([index[0], size[0]], value);
      AdjacencyMatrixCopy.set(index, 0);
      
      AdjacencyMatrixCopy.set([size[0]+1, index[0]], AdjacencyMatrixCopy.get(diag));
      AdjacencyMatrixCopy.set([index[1], size[0]+1], AdjacencyMatrixCopy.get(diag));
      AdjacencyMatrixCopy.set(index, 0);
    }
  });

  const add = (path: number[], volume: number) => {
    path.forEach((from, i) => {
      if (i === path.length - 1) {
        return;
      }
      const to = path[i+1];

      AdjacencyMatrixCopy.set([from, to], AdjacencyMatrixCopy.get([from, to]) - volume);
      AdjacencyMatrixCopy.set([to, from], AdjacencyMatrixCopy.get([to, from]) + volume);

      if (AdjacencyMatrixCopy.get([from, to]) < 0) {
        throw new Error(`フローの容量が負の値になりました。from: ${from}, to: ${to}, capacity: ${AdjacencyMatrixCopy.get([from, to])}`);
      }
    });
  }

  return {
    vSize: AdjacencyMatrix.size()[0],
    capacity: (i: number, j: number) => AdjacencyMatrixCopy.get([i, j]),
    add: add
  }
}