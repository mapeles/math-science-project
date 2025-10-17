"use client";

import { useState } from 'react';

const NeuralNetworkDiagram = () => {
  // 신경망 값들을 상태로 관리
  const [inputValues, setInputValues] = useState({ x1: 1.0, x2: 0.5 });
  
  const [weights1, setWeights1] = useState({
    w11: 0.5, w21: -0.3, w31: 0.8,
    w12: 0.2, w22: 0.6, w32: -0.4
  });
  
  const [weights2, setWeights2] = useState({
    w11: 0.7, w21: -0.2,
    w12: 0.3, w22: 0.9,
    w13: -0.5, w23: 0.4
  });

  // 선택된 요소 추적
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedWeight, setSelectedWeight] = useState(null);

  // 노드 선택 핸들러 (가중치 선택 해제)
  const handleNodeClick = (nodeId) => {
    setSelectedNode(nodeId);
    setSelectedWeight(null);
  };

  // 가중치 선택 핸들러 (노드 선택 해제)
  const handleWeightClick = (weightId) => {
    setSelectedWeight(weightId);
    setSelectedNode(null);
  };

  // 값 업데이트 함수들
  const updateInputValue = (key, value) => {
    setInputValues(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const updateWeight1 = (key, value) => {
    setWeights1(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const updateWeight2 = (key, value) => {
    setWeights2(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  // 역전파 정보 계산
  const getBackpropInfo = (selectedInfo) => {
    if (!selectedInfo) return null;

    if (selectedInfo.type === 'weight') {
      const { id, from, to, value } = selectedInfo.data;
      
      // W(1)_nm: y1, y2에 대한 영향을 W(2)_jn을 통해 계산
      if (id.startsWith('w1_')) {
        const sub = id.slice(3); // "11", "21" 등
        const n = sub[0]; // 첫 번째 인덱스 (hidden 노드)
        
        // y1에 대한 영향: W(2)_1n 값
        const w2_1n_key = `w${1}${n}`;
        const w2_1n_value = weights2[w2_1n_key];
        
        // y2에 대한 영향: W(2)_2n 값
        const w2_2n_key = `w${2}${n}`;
        const w2_2n_value = weights2[w2_2n_key];
        
        return {
          y1: {
            derivative: w2_1n_value,
            weightLabel: `W⁽²⁾₁${n}`,
            effect: w2_1n_value > 0 ? '증가' : '감소'
          },
          y2: {
            derivative: w2_2n_value,
            weightLabel: `W⁽²⁾₂${n}`,
            effect: w2_2n_value > 0 ? '증가' : '감소'
          }
        };
      }
      
      // W(2)_nm: y1, y2에 대한 영향을 a(1)_m을 통해 계산
      if (id.startsWith('w2_')) {
        const sub = id.slice(3); // "11", "21" 등
        const outputIdx = sub[0]; // 출력 노드 인덱스 (1 or 2)
        const hiddenIdx = sub[1]; // hidden 노드 인덱스
        
        const hiddenKey = `a${hiddenIdx}`;
        const hiddenValue = hiddenValues[hiddenKey];
        
        if (outputIdx === '1') {
          return {
            y1: {
              derivative: hiddenValue,
              weightLabel: `a⁽¹⁾${hiddenIdx}`,
              effect: hiddenValue > 0 ? '증가' : '감소'
            },
            y2: null
          };
        } else {
          return {
            y1: null,
            y2: {
              derivative: hiddenValue,
              weightLabel: `a⁽¹⁾${hiddenIdx}`,
              effect: hiddenValue > 0 ? '증가' : '감소'
            }
          };
        }
      }
    }
    
    return null;
  };

  // 선택된 요소의 정보 가져오기
  const getSelectedInfo = () => {
    if (selectedNode) {
      const node = [...layers.input, ...layers.hidden, ...layers.output].find(n => n.id === selectedNode);
      return { type: 'node', data: node };
    }
    if (selectedWeight) {
      const weight = connections.find(w => w.id === selectedWeight);
      return { type: 'weight', data: weight };
    }
    return null;
  };

  // Hidden layer 값 계산 (간단한 선형 변환, 실제로는 활성화 함수 적용)
  const hiddenValues = {
    a1: inputValues.x1 * weights1.w11 + inputValues.x2 * weights1.w12,
    a2: inputValues.x1 * weights1.w21 + inputValues.x2 * weights1.w22,
    a3: inputValues.x1 * weights1.w31 + inputValues.x2 * weights1.w32,
  };

  // Output layer 값 계산
  const outputValues = {
    y1: hiddenValues.a1 * weights2.w11 + hiddenValues.a2 * weights2.w12 + hiddenValues.a3 * weights2.w13,
    y2: hiddenValues.a1 * weights2.w21 + hiddenValues.a2 * weights2.w22 + hiddenValues.a3 * weights2.w23,
  };

  // 각 레이어의 노드 위치를 정의합니다.
  const layers = {
    input: [
      { id: 'x1', cx: 100, cy: 150, base: 'x', sub: '1', value: inputValues.x1 },
      { id: 'x2', cx: 100, cy: 350, base: 'x', sub: '2', value: inputValues.x2 },
    ],
    hidden: [
      { id: 'a1', cx: 400, cy: 100, base: 'a', super: '(1)', sub: '1', value: hiddenValues.a1 },
      { id: 'a2', cx: 400, cy: 250, base: 'a', super: '(1)', sub: '2', value: hiddenValues.a2 },
      { id: 'a3', cx: 400, cy: 400, base: 'a', super: '(1)', sub: '3', value: hiddenValues.a3 },
    ],
    output: [
      { id: 'y1', cx: 700, cy: 150, base: 'y', sub: '1', value: outputValues.y1 },
      { id: 'y2', cx: 700, cy: 350, base: 'y', sub: '2', value: outputValues.y2 },
    ],
  };

  // 노드 간의 연결(가중치)을 정의합니다.
  const connections = [
    // Input -> Hidden
    { id: 'w1_11', from: 'x1', to: 'a1', base: 'W', super: '(1)', sub: '11', value: weights1.w11 },
    { id: 'w1_21', from: 'x1', to: 'a2', base: 'W', super: '(1)', sub: '21', value: weights1.w21 },
    { id: 'w1_31', from: 'x1', to: 'a3', base: 'W', super: '(1)', sub: '31', value: weights1.w31 },
    { id: 'w1_12', from: 'x2', to: 'a1', base: 'W', super: '(1)', sub: '12', value: weights1.w12 },
    { id: 'w1_22', from: 'x2', to: 'a2', base: 'W', super: '(1)', sub: '22', value: weights1.w22 },
    { id: 'w1_32', from: 'x2', to: 'a3', base: 'W', super: '(1)', sub: '32', value: weights1.w32 },
    // Hidden -> Output
    { id: 'w2_11', from: 'a1', to: 'y1', base: 'W', super: '(2)', sub: '11', value: weights2.w11 },
    { id: 'w2_21', from: 'a1', to: 'y2', base: 'W', super: '(2)', sub: '21', value: weights2.w21 },
    { id: 'w2_12', from: 'a2', to: 'y1', base: 'W', super: '(2)', sub: '12', value: weights2.w12 },
    { id: 'w2_22', from: 'a2', to: 'y2', base: 'W', super: '(2)', sub: '22', value: weights2.w22 },
    { id: 'w2_13', from: 'a3', to: 'y1', base: 'W', super: '(2)', sub: '13', value: weights2.w13 },
    { id: 'w2_23', from: 'a3', to: 'y2', base: 'W', super: '(2)', sub: '23', value: weights2.w23 },
  ];

  // 수식 레이블을 렌더링하는 헬퍼 함수
  const renderLabel = (item, x, y, fontSize = 20, isSelected = false, showValue = false) => {
    const color = isSelected ? 'red' : 'black';
    const displayText = showValue ? item.value.toFixed(2) : null;
    
    return (
      <text x={x} y={y} fontSize={fontSize} textAnchor="middle" fill={color}>
        {showValue ? (
          <tspan>{displayText}</tspan>
        ) : (
          <>
            <tspan>{item.base}</tspan>
            {item.super && (
              <tspan fontSize={fontSize * 0.6} dy={-fontSize * 0.35}>
                {item.super}
              </tspan>
            )}
            {item.sub && (
              <tspan fontSize={fontSize * 0.6} dy={item.super ? fontSize * 0.7 : fontSize * 0.35}>
                {item.sub}
              </tspan>
            )}
          </>
        )}
      </text>
    );
  };

  // 다이어그램 렌더링 함수 (수식 또는 실제 값)
  const renderDiagram = (showValue = false, offsetX = 0) => {
    return (
      <g transform={`translate(${offsetX}, 0)`}>
        {/* 연결선과 가중치 텍스트 렌더링 */}
        {connections.map((connection, index) => {
          const fromNode = [...layers.input, ...layers.hidden].find(n => n.id === connection.from);
          const toNode = [...layers.hidden, ...layers.output].find(n => n.id === connection.to);
          
          // 라벨 위치 계산 (선의 중앙)
          const textX = (fromNode.cx + toNode.cx) / 2;
          const textY = (fromNode.cy + toNode.cy) / 2 - 10;
          
          // 선의 각도 계산
          const angle = Math.atan2(toNode.cy - fromNode.cy, toNode.cx - fromNode.cx) * (180 / Math.PI);
          const isSelected = selectedWeight === connection.id;

          return (
            <g key={index}>
              <line
                x1={fromNode.cx}
                y1={fromNode.cy}
                x2={toNode.cx}
                y2={toNode.cy}
                stroke="gray"
                strokeWidth="1"
              />
              <g 
                transform={`rotate(${angle}, ${textX}, ${textY})`}
                onClick={(e) => {
                  e.stopPropagation();
                  handleWeightClick(connection.id);
                }}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={textX - 30}
                  y={textY - 12}
                  width="60"
                  height="24"
                  fill="transparent"
                  style={{ pointerEvents: 'all' }}
                />
                <g style={{ pointerEvents: 'none' }}>
                  {renderLabel(connection, textX, textY, 16, isSelected, showValue)}
                </g>
              </g>
            </g>
          );
        })}

        {/* 노드 (동그라미와 라벨) 렌더링 */}
        {Object.values(layers).flat().map(node => {
          const isSelected = selectedNode === node.id;
          return (
            <g 
              key={node.id}
              onClick={() => handleNodeClick(node.id)}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={node.cx}
                cy={node.cy}
                r="40"
                stroke="black"
                strokeWidth="2"
                fill="white"
                style={{ pointerEvents: 'all' }}
              />
              <g style={{ pointerEvents: 'none' }}>
                {renderLabel(node, node.cx, node.cy + 5, 20, isSelected, showValue)}
              </g>
            </g>
          );
        })}
      </g>
    );
  };

  const selectedInfo = getSelectedInfo();

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-4">
        {/* 왼쪽: 수식 다이어그램 */}
        <div className="flex-1">
          <h2 className="text-lg font-bold mb-2 text-center">신경망 구조 (수식)</h2>
          <svg viewBox="0 0 800 500" style={{ width: '100%', height: 'auto', fontFamily: 'serif' }}>
            {renderDiagram(false, 0)}
          </svg>
        </div>

        {/* 오른쪽: 실제 값 다이어그램 */}
        <div className="flex-1">
          <h2 className="text-lg font-bold mb-2 text-center">신경망 구조 (실제 값)</h2>
          <svg viewBox="0 0 800 500" style={{ width: '100%', height: 'auto', fontFamily: 'serif' }}>
            {renderDiagram(true, 0)}
          </svg>
        </div>
      </div>

      {/* 설정 패널 */}
      {selectedInfo && (
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6 transition-all duration-300 ease-in-out">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-2xl font-bold text-gray-800">
              {selectedInfo.type === 'node' ? '노드 설정' : '가중치 설정'}
            </h3>
            <button
              onClick={() => {
                setSelectedNode(null);
                setSelectedWeight(null);
              }}
              className="text-gray-500 hover:text-gray-700 transition-colors duration-200"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="flex gap-6">
            {/* 왼쪽: 설정 영역 */}
            <div className="flex-1">
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-4 mb-4">
                <div className="text-sm text-gray-600 mb-2">선택된 요소</div>
                <div className="text-lg font-semibold text-gray-800">
                  {selectedInfo.type === 'node' && (
                    <span>
                      {selectedInfo.data.base}
                      {selectedInfo.data.super && <sup className="text-sm">{selectedInfo.data.super}</sup>}
                      {selectedInfo.data.sub && <sub className="text-sm">{selectedInfo.data.sub}</sub>}
                    </span>
                  )}
                  {selectedInfo.type === 'weight' && (
                    <span>
                      {selectedInfo.data.base}
                      {selectedInfo.data.super && <sup className="text-sm">{selectedInfo.data.super}</sup>}
                      {selectedInfo.data.sub && <sub className="text-sm">{selectedInfo.data.sub}</sub>}
                    </span>
                  )}
                </div>
              </div>

              {/* 입력 노드 설정 */}
              {selectedInfo.type === 'node' && selectedInfo.data.id.startsWith('x') && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-3">
                      입력 값 선택
                    </label>
                    <div className="flex gap-3">
                      <button
                        onClick={() => updateInputValue(selectedInfo.data.id, 0)}
                        className={`flex-1 py-4 px-6 rounded-lg font-bold text-lg transition-all duration-200 ${
                          selectedInfo.data.value === 0
                            ? 'bg-blue-500 text-white shadow-lg scale-105'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        0
                      </button>
                      <button
                        onClick={() => updateInputValue(selectedInfo.data.id, 1)}
                        className={`flex-1 py-4 px-6 rounded-lg font-bold text-lg transition-all duration-200 ${
                          selectedInfo.data.value === 1
                            ? 'bg-blue-500 text-white shadow-lg scale-105'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`}
                      >
                        1
                      </button>
                    </div>
                  </div>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <div className="text-sm text-blue-800 text-center">
                      현재 값: <span className="font-bold text-xl">{selectedInfo.data.value}</span>
                    </div>
                  </div>
                </div>
              )}

              {/* 가중치 설정 */}
              {selectedInfo.type === 'weight' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      가중치 값 조절
                    </label>
                    <div className="space-y-3">
                      <input
                        type="range"
                        min="-1"
                        max="1"
                        step="0.01"
                        value={selectedInfo.data.value}
                        onChange={(e) => {
                          const id = selectedInfo.data.id;
                          if (id.startsWith('w1_')) {
                            const key = 'w' + id.slice(3);
                            updateWeight1(key, e.target.value);
                          } else if (id.startsWith('w2_')) {
                            const key = 'w' + id.slice(3);
                            updateWeight2(key, e.target.value);
                          }
                        }}
                        className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-500"
                        style={{
                          background: `linear-gradient(to right, 
                            #ef4444 0%, 
                            #fbbf24 50%, 
                            #22c55e 100%)`
                        }}
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>-1.0</span>
                        <span>0.0</span>
                        <span>1.0</span>
                      </div>
                    </div>
                  </div>
                  <div 
                    className="rounded-lg p-3 border-2 transition-all duration-200"
                    style={{
                      backgroundColor: (() => {
                        const val = selectedInfo.data.value;
                        if (val < 0) {
                          // -1 ~ 0: 빨간색에서 노란색으로
                          const ratio = (val + 1); // 0 ~ 1
                          return `rgb(${239 - ratio * (239 - 251)}, ${68 + ratio * (187 - 68)}, ${68 + ratio * (20 - 68)})`;
                        } else {
                          // 0 ~ 1: 노란색에서 초록색으로
                          const ratio = val; // 0 ~ 1
                          return `rgb(${251 - ratio * (251 - 34)}, ${187 + ratio * (197 - 187)}, ${20 + ratio * (94 - 20)})`;
                        }
                      })(),
                      borderColor: (() => {
                        const val = selectedInfo.data.value;
                        if (val < -0.3) return '#ef4444';
                        if (val < 0.3) return '#fbbf24';
                        return '#22c55e';
                      })(),
                      color: Math.abs(selectedInfo.data.value) > 0.5 ? '#ffffff' : '#1f2937'
                    }}
                  >
                    <div className="text-sm font-medium text-center">
                      현재 값: <span className="font-bold text-2xl">{selectedInfo.data.value.toFixed(3)}</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 text-center mt-2">
                    연결: {selectedInfo.data.from} → {selectedInfo.data.to}
                  </div>
                </div>
              )}

              {/* Hidden/Output 노드 정보 (읽기 전용) */}
              {selectedInfo.type === 'node' && !selectedInfo.data.id.startsWith('x') && (
                <div className="space-y-4">
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="text-sm text-purple-700 mb-2">계산된 값 (읽기 전용)</div>
                    <div className="text-2xl font-bold text-purple-900">
                      {selectedInfo.data.value.toFixed(4)}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    이 값은 입력 값과 가중치에 의해 자동으로 계산됩니다.
                  </div>
                </div>
              )}
            </div>

            {/* 오른쪽: 역전파 정보 */}
            {selectedInfo.type === 'weight' && (() => {
              const backpropInfo = getBackpropInfo(selectedInfo);
              return backpropInfo && (
                <div className="flex-1 border-l border-gray-200 pl-6">
                  <h4 className="text-lg font-bold text-gray-800 mb-4">역전파 정보</h4>
                  
                  <div className="flex gap-4">
                    {/* y1에 대한 영향 */}
                    {backpropInfo.y1 && (
                      <div className="flex-1 bg-gradient-to-br from-pink-50 to-rose-50 rounded-xl p-4">
                        <div className="text-sm font-medium text-gray-700 mb-2">y₁에 대한 영향</div>
                        <div className="space-y-2">
                          <div>
                            <div className="text-xs text-gray-600">미분값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y1.weightLabel}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-600">현재 값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y1.derivative.toFixed(3)}</div>
                          </div>
                          <div className={`mt-2 p-2 rounded-lg ${
                            backpropInfo.y1.derivative > 0 
                              ? 'bg-green-100 border border-green-300' 
                              : 'bg-red-100 border border-red-300'
                          }`}>
                            <div className="text-xs font-medium">
                              {backpropInfo.y1.derivative > 0 ? '✓ ' : '✗ '}
                              가중치 증가 시 y₁
                              <span className="font-bold ml-1">
                                {backpropInfo.y1.effect}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* y2에 대한 영향 */}
                    {backpropInfo.y2 && (
                      <div className="flex-1 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-4">
                        <div className="text-sm font-medium text-gray-700 mb-2">y₂에 대한 영향</div>
                        <div className="space-y-2">
                          <div>
                            <div className="text-xs text-gray-600">미분값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y2.weightLabel}</div>
                          </div>
                          <div>
                            <div className="text-xs text-gray-600">현재 값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y2.derivative.toFixed(3)}</div>
                          </div>
                          <div className={`mt-2 p-2 rounded-lg ${
                            backpropInfo.y2.derivative > 0 
                              ? 'bg-green-100 border border-green-300' 
                              : 'bg-red-100 border border-red-300'
                          }`}>
                            <div className="text-xs font-medium">
                              {backpropInfo.y2.derivative > 0 ? '✓ ' : '✗ '}
                              가중치 증가 시 y₂
                              <span className="font-bold ml-1">
                                {backpropInfo.y2.effect}
                              </span>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <div className="text-xs text-gray-600">
                      <strong>역전파 원리:</strong> W⁽¹⁾는 W⁽²⁾를 통해, W⁽²⁾는 a⁽¹⁾에 비례하여 y에 영향을 미칩니다.
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
};


export default function Home() {
  return (
    <div className="p-4">
      <h1 className="text-4xl font-bold mt-3 text-center">신경망 시각화</h1>
      <NeuralNetworkDiagram/>
    </div>
  );
}
