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
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  입력 값
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={selectedInfo.data.value}
                  onChange={(e) => updateInputValue(selectedInfo.data.id, e.target.value)}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="text-sm text-blue-800">
                  현재 값: <span className="font-semibold">{selectedInfo.data.value.toFixed(3)}</span>
                </div>
              </div>
            </div>
          )}

          {/* 가중치 설정 */}
          {selectedInfo.type === 'weight' && (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  가중치 값
                </label>
                <input
                  type="number"
                  step="0.1"
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
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="text-sm text-green-800">
                  현재 값: <span className="font-semibold">{selectedInfo.data.value.toFixed(3)}</span>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-2">
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
