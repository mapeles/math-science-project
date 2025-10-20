"use client";

import { useState } from 'react';

const NeuralNetworkDiagram = () => {
  // 신경망 값들을 상태로 관리
  const [inputValues, setInputValues] = useState({ x1: 1.0, x2: 0 });
  
  // Hidden Layer 가중치
  // a1: OR 게이트 (둘 중 하나라도 1이면 활성화)
  // a2: AND 게이트 (둘 다 1이어야 활성화)
  // a3: 사용 안 함
  const [weights1, setWeights1] = useState({
    w11: 20.0,  w21: 20.0,  w31: 0.0,   // x1 → hidden
    w12: 20.0,  w22: 20.0,  w32: 0.0    // x2 → hidden
  });
  
  // Output Layer 가중치
  // y1 (출력=1): a1 강화, a2 억제
  // y2 (출력=0): a1 억제, a2 강화
  const [weights2, setWeights2] = useState({
    w11: 20.0,  w21: -20.0,  // a1 → output
    w12: -20.0, w22: 20.0,   // a2 → output
    w13: 0.0,   w23: 0.0     // a3 → output
  });

  // 편향(bias)
  const [biases1, setBiases1] = useState({
    b1: -10.0,  // OR: 하나만 있어도 활성화
    b2: -30.0,  // AND: 둘 다 있어야 활성화
    b3: 0.0     // 사용 안 함
  });

  const [biases2, setBiases2] = useState({
    b1: -10.0,  // y1 조정
    b2: 10.0    // y2 조정
  });

  // 선택된 요소 추적
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedWeight, setSelectedWeight] = useState(null);
  const [selectedBias, setSelectedBias] = useState(null);

  // Sigmoid 활성화 함수
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const sigmoidDerivative = (sigmoidOutput) => sigmoidOutput * (1 - sigmoidOutput);

  // 노드 선택 핸들러 (가중치, 편향 선택 해제)
  const handleNodeClick = (nodeId) => {
    setSelectedNode(nodeId);
    setSelectedWeight(null);
    setSelectedBias(null);
  };

  // 가중치 선택 핸들러 (노드, 편향 선택 해제)
  const handleWeightClick = (weightId) => {
    setSelectedWeight(weightId);
    setSelectedNode(null);
    setSelectedBias(null);
  };

  // 편향 선택 핸들러
  const handleBiasClick = (biasId) => {
    setSelectedBias(biasId);
    setSelectedNode(null);
    setSelectedWeight(null);
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

  const updateBias1 = (key, value) => {
    setBiases1(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  const updateBias2 = (key, value) => {
    setBiases2(prev => ({ ...prev, [key]: parseFloat(value) || 0 }));
  };

  // 역전파 정보 계산
  const getBackpropInfo = (selectedInfo) => {
    if (!selectedInfo) return null;

    if (selectedInfo.type === 'weight') {
      const { id } = selectedInfo.data;
      
      // W(1)_nm: 체인 룰 두 번 적용
      if (id.startsWith('w1_')) {
        const sub = id.slice(3); // "11", "21" 등
        const n = parseInt(sub[0]); // hidden 노드 인덱스 (1,2,3)
        const m = parseInt(sub[1]); // input 인덱스 (1,2)
        
        const hiddenKey = `a${n}`;
        const hiddenValue = hiddenValues[hiddenKey];
        const hiddenGrad = sigmoidDerivative(hiddenValue);
        
        const inputKey = `x${m}`;
        const inputValue = inputValues[inputKey];
        
        // y1에 대한 영향
        const w2_1n_key = `w${1}${n}`;
        const w2_1n_value = weights2[w2_1n_key];
        const y1_value = outputValues.y1;
        const y1_grad = sigmoidDerivative(y1_value);
        
        const grad_y1 = y1_grad * w2_1n_value * hiddenGrad * inputValue;
        
        // y2에 대한 영향
        const w2_2n_key = `w${2}${n}`;
        const w2_2n_value = weights2[w2_2n_key];
        const y2_value = outputValues.y2;
        const y2_grad = sigmoidDerivative(y2_value);
        
        const grad_y2 = y2_grad * w2_2n_value * hiddenGrad * inputValue;
        
        return {
          y1: {
            derivative: grad_y1,
            formula: `y₁(1-y₁) · W⁽²⁾₁${n} · a⁽¹⁾${n}(1-a⁽¹⁾${n}) · x${m}`,
            components: {
              outputGrad: y1_grad.toFixed(3),
              weight2: w2_1n_value.toFixed(3),
              hiddenGrad: hiddenGrad.toFixed(3),
              input: inputValue.toFixed(3)
            },
            effect: grad_y1 > 0 ? '증가' : '감소'
          },
          y2: {
            derivative: grad_y2,
            formula: `y₂(1-y₂) · W⁽²⁾₂${n} · a⁽¹⁾${n}(1-a⁽¹⁾${n}) · x${m}`,
            components: {
              outputGrad: y2_grad.toFixed(3),
              weight2: w2_2n_value.toFixed(3),
              hiddenGrad: hiddenGrad.toFixed(3),
              input: inputValue.toFixed(3)
            },
            effect: grad_y2 > 0 ? '증가' : '감소'
          }
        };
      }
      
      // W(2)_jm: 체인 룰 한 번 적용
      if (id.startsWith('w2_')) {
        const sub = id.slice(3); // "11", "21" 등
        const j = parseInt(sub[0]); // 출력 노드 인덱스 (1 or 2)
        const m = parseInt(sub[1]); // hidden 노드 인덱스 (1,2,3)
        
        const hiddenKey = `a${m}`;
        const hiddenValue = hiddenValues[hiddenKey];
        
        if (j === 1) {
          const y1_value = outputValues.y1;
          const y1_grad = sigmoidDerivative(y1_value);
          const grad_y1 = y1_grad * hiddenValue;
          
          return {
            y1: {
              derivative: grad_y1,
              formula: `y₁(1-y₁) · a⁽¹⁾${m}`,
              components: {
                outputGrad: y1_grad.toFixed(3),
                hidden: hiddenValue.toFixed(3)
              },
              effect: grad_y1 > 0 ? '증가' : '감소'
            },
            y2: null
          };
        } else {
          const y2_value = outputValues.y2;
          const y2_grad = sigmoidDerivative(y2_value);
          const grad_y2 = y2_grad * hiddenValue;
          
          return {
            y1: null,
            y2: {
              derivative: grad_y2,
              formula: `y₂(1-y₂) · a⁽¹⁾${m}`,
              components: {
                outputGrad: y2_grad.toFixed(3),
                hidden: hiddenValue.toFixed(3)
              },
              effect: grad_y2 > 0 ? '증가' : '감소'
            }
          };
        }
      }
    }
    
    // 편향에 대한 미분
    if (selectedInfo.type === 'bias') {
      const { id } = selectedInfo.data;
      
      // b(2)_j: Output 편향
      if (id.includes('_out')) {
        const j = parseInt(id.match(/b(\d+)/)[1]);
        
        if (j === 1) {
          const y1_value = outputValues.y1;
          const grad = sigmoidDerivative(y1_value);
          
          return {
            y1: {
              derivative: grad,
              formula: `y₁(1-y₁)`,
              components: { outputGrad: grad.toFixed(3) },
              effect: '증가'
            },
            y2: null
          };
        } else {
          const y2_value = outputValues.y2;
          const grad = sigmoidDerivative(y2_value);
          
          return {
            y1: null,
            y2: {
              derivative: grad,
              formula: `y₂(1-y₂)`,
              components: { outputGrad: grad.toFixed(3) },
              effect: '증가'
            }
          };
        }
      }
      
      // b(1)_n: Hidden 편향
      const n = parseInt(id.match(/b(\d+)/)[1]);
      const hiddenKey = `a${n}`;
      const hiddenValue = hiddenValues[hiddenKey];
      const hiddenGrad = sigmoidDerivative(hiddenValue);
      
      // y1, y2 각각에 대한 영향
      const w2_1n = weights2[`w1${n}`];
      const w2_2n = weights2[`w2${n}`];
      
      const y1_grad = sigmoidDerivative(outputValues.y1);
      const y2_grad = sigmoidDerivative(outputValues.y2);
      
      return {
        y1: {
          derivative: y1_grad * w2_1n * hiddenGrad,
          formula: `y₁(1-y₁) · W⁽²⁾₁${n} · a⁽¹⁾${n}(1-a⁽¹⁾${n})`,
          components: {
            outputGrad: y1_grad.toFixed(3),
            weight2: w2_1n.toFixed(3),
            hiddenGrad: hiddenGrad.toFixed(3)
          },
          effect: (y1_grad * w2_1n * hiddenGrad) > 0 ? '증가' : '감소'
        },
        y2: {
          derivative: y2_grad * w2_2n * hiddenGrad,
          formula: `y₂(1-y₂) · W⁽²⁾₂${n} · a⁽¹⁾${n}(1-a⁽¹⁾${n})`,
          components: {
            outputGrad: y2_grad.toFixed(3),
            weight2: w2_2n.toFixed(3),
            hiddenGrad: hiddenGrad.toFixed(3)
          },
          effect: (y2_grad * w2_2n * hiddenGrad) > 0 ? '증가' : '감소'
        }
      };
    }
    
    return null;
  };

  // 선택된 요소의 정보 가져오기
  const getSelectedInfo = () => {
    if (selectedNode) {
      const node = [...layers.input, ...layers.hidden, ...layers.output, ...(layers.bias1 || []), ...(layers.bias2 || [])].find(n => n.id === selectedNode);
      return { type: 'node', data: node };
    }
    if (selectedWeight) {
      const weight = connections.find(w => w.id === selectedWeight);
      return { type: 'weight', data: weight };
    }
    if (selectedBias) {
      const bias = [...(layers.bias1 || []), ...(layers.bias2 || [])].find(b => b.id === selectedBias);
      return { type: 'bias', data: bias };
    }
    return null;
  };

  // Hidden layer 값 계산 (Sigmoid 활성화 함수 적용)
  const hiddenZValues = {
    z1: inputValues.x1 * weights1.w11 + inputValues.x2 * weights1.w12 + biases1.b1,
    z2: inputValues.x1 * weights1.w21 + inputValues.x2 * weights1.w22 + biases1.b2,
    z3: inputValues.x1 * weights1.w31 + inputValues.x2 * weights1.w32 + biases1.b3,
  };

  const hiddenValues = {
    a1: sigmoid(hiddenZValues.z1),
    a2: sigmoid(hiddenZValues.z2),
    a3: sigmoid(hiddenZValues.z3),
  };

  // Output layer 값 계산 (Sigmoid 활성화 함수 적용)
  const outputZValues = {
    z1: hiddenValues.a1 * weights2.w11 + hiddenValues.a2 * weights2.w12 + hiddenValues.a3 * weights2.w13 + biases2.b1,
    z2: hiddenValues.a1 * weights2.w21 + hiddenValues.a2 * weights2.w22 + hiddenValues.a3 * weights2.w23 + biases2.b2,
  };

  const outputValues = {
    y1: sigmoid(outputZValues.z1),
    y2: sigmoid(outputZValues.z2),
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
    bias1: [
      { id: 'b1', cx: 320, cy: 50, base: 'b', super: '(1)', sub: '1', value: biases1.b1 },
      { id: 'b2', cx: 320, cy: 200, base: 'b', super: '(1)', sub: '2', value: biases1.b2 },
      { id: 'b3', cx: 320, cy: 350, base: 'b', super: '(1)', sub: '3', value: biases1.b3 },
    ],
    bias2: [
      { id: 'b1_out', cx: 620, cy: 100, base: 'b', super: '(2)', sub: '1', value: biases2.b1 },
      { id: 'b2_out', cx: 620, cy: 300, base: 'b', super: '(2)', sub: '2', value: biases2.b2 },
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

  // 편향 연결선 정의
  const biasConnections = [
    // Bias1 -> Hidden
    { id: 'bias1_1', from: 'b1', to: 'a1' },
    { id: 'bias1_2', from: 'b2', to: 'a2' },
    { id: 'bias1_3', from: 'b3', to: 'a3' },
    // Bias2 -> Output
    { id: 'bias2_1', from: 'b1_out', to: 'y1' },
    { id: 'bias2_2', from: 'b2_out', to: 'y2' },
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

        {/* 편향 연결선 렌더링 */}
        {biasConnections.map((connection, index) => {
          const fromNode = [...(layers.bias1 || []), ...(layers.bias2 || [])].find(n => n.id === connection.from);
          const toNode = [...layers.hidden, ...layers.output].find(n => n.id === connection.to);
          
          return (
            <g key={`bias-${index}`}>
              <line
                x1={fromNode.cx}
                y1={fromNode.cy}
                x2={toNode.cx}
                y2={toNode.cy}
                stroke="orange"
                strokeWidth="2"
                strokeDasharray="5,5"
                opacity="0.6"
              />
            </g>
          );
        })}

        {/* 노드 (동그라미와 라벨) 렌더링 */}
        {Object.values(layers).flat().map(node => {
          const isSelected = selectedNode === node.id || selectedBias === node.id;
          const isBias = node.id.startsWith('b');
          
          return (
            <g 
              key={node.id}
              onClick={() => isBias ? handleBiasClick(node.id) : handleNodeClick(node.id)}
              style={{ cursor: 'pointer' }}
            >
              <circle
                cx={node.cx}
                cy={node.cy}
                r={isBias ? "30" : "40"}
                stroke={isBias ? "orange" : "black"}
                strokeWidth="2"
                fill={isBias ? "#fff3cd" : "white"}
                style={{ pointerEvents: 'all' }}
              />
              <g style={{ pointerEvents: 'none' }}>
                {renderLabel(node, node.cx, node.cy + 5, isBias ? 16 : 20, isSelected, showValue)}
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
              {selectedInfo.type === 'node' ? '노드 설정' : selectedInfo.type === 'bias' ? '편향 설정' : '가중치 설정'}
            </h3>
            <button
              onClick={() => {
                setSelectedNode(null);
                setSelectedWeight(null);
                setSelectedBias(null);
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
                  {selectedInfo.type === 'bias' && (
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
                        min="-20"
                        max="20"
                        step="0.05"
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
                        <span>-20.0</span>
                        <span>0.0</span>
                        <span>20.0</span>
                      </div>
                    </div>
                  </div>
                  <div 
                    className="rounded-lg p-3 border-2 transition-all duration-200"
                    style={{
                      backgroundColor: (() => {
                        const val = selectedInfo.data.value;
                        const maxVal = 20;
                        if (val < 0) {
                          // -20 ~ 0: 빨간색에서 노란색으로
                          const ratio = (val + maxVal) / maxVal; // 0 ~ 1
                          return `rgb(${239 - ratio * (239 - 251)}, ${68 + ratio * (187 - 68)}, ${68 + ratio * (20 - 68)})`;
                        } else {
                          // 0 ~ 20: 노란색에서 초록색으로
                          const ratio = val / maxVal; // 0 ~ 1
                          return `rgb(${251 - ratio * (251 - 34)}, ${187 + ratio * (197 - 187)}, ${20 + ratio * (94 - 20)})`;
                        }
                      })(),
                      borderColor: (() => {
                        const val = selectedInfo.data.value;
                        if (val < -5) return '#ef4444';
                        if (val < 5) return '#fbbf24';
                        return '#22c55e';
                      })(),
                      color: Math.abs(selectedInfo.data.value) > 10 ? '#ffffff' : '#1f2937'
                    }}
                  >
                    <div className="text-sm font-medium text-center">
                      현재 값: <span className="font-bold text-2xl">{selectedInfo.data.value.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 text-center mt-2">
                    연결: {selectedInfo.data.from} → {selectedInfo.data.to}
                  </div>
                </div>
              )}

              {/* Hidden/Output 노드 정보 (읽기 전용) */}
              {selectedInfo.type === 'node' && !selectedInfo.data.id.startsWith('x') && !selectedInfo.data.id.startsWith('b') && (
                <div className="space-y-4">
                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="text-sm text-purple-700 mb-2">계산된 값 (읽기 전용)</div>
                    <div className="text-2xl font-bold text-purple-900">
                      {selectedInfo.data.value.toFixed(4)}
                    </div>
                  </div>
                  <div className="text-xs text-gray-500">
                    이 값은 입력 값과 가중치, 편향에 의해 Sigmoid 함수를 통해 자동으로 계산됩니다.
                  </div>
                </div>
              )}

              {/* 편향 설정 */}
              {selectedInfo.type === 'bias' && (
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      편향 값 조절
                    </label>
                    <div className="space-y-3">
                      <input
                        type="range"
                        min="-30"
                        max="30"
                        step="0.05"
                        value={selectedInfo.data.value}
                        onChange={(e) => {
                          const id = selectedInfo.data.id;
                          if (id.includes('_out')) {
                            const key = 'b' + id.match(/b(\d+)/)[1];
                            updateBias2(key, e.target.value);
                          } else {
                            const key = id;
                            updateBias1(key, e.target.value);
                          }
                        }}
                        className="w-full h-3 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500"
                      />
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>-30.0</span>
                        <span>0.0</span>
                        <span>30.0</span>
                      </div>
                    </div>
                  </div>
                  <div className="bg-orange-50 border-2 border-orange-300 rounded-lg p-3">
                    <div className="text-sm font-medium text-center text-orange-900">
                      현재 값: <span className="font-bold text-2xl">{selectedInfo.data.value.toFixed(1)}</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 text-center">
                    편향은 활성화 임계값을 조절합니다
                  </div>
                </div>
              )}
            </div>

            {/* 오른쪽: 역전파 정보 */}
            {(selectedInfo.type === 'weight' || selectedInfo.type === 'bias') && (() => {
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
                            <div className="text-xs text-gray-600">미분 공식:</div>
                            <div className="font-mono text-xs">{backpropInfo.y1.formula}</div>
                          </div>
                          {backpropInfo.y1.components && (
                            <div className="text-xs bg-white rounded p-2">
                              {Object.entries(backpropInfo.y1.components).map(([key, val]) => (
                                <div key={key}>{key}: {val}</div>
                              ))}
                            </div>
                          )}
                          <div>
                            <div className="text-xs text-gray-600">미분값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y1.derivative.toFixed(4)}</div>
                          </div>
                          <div className={`mt-2 p-2 rounded-lg ${
                            backpropInfo.y1.derivative > 0 
                              ? 'bg-green-100 border border-green-300' 
                              : 'bg-red-100 border border-red-300'
                          }`}>
                            <div className="text-xs font-medium">
                              {backpropInfo.y1.derivative > 0 ? '✓ ' : '✗ '}
                              증가 시 y₁
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
                            <div className="text-xs text-gray-600">미분 공식:</div>
                            <div className="font-mono text-xs">{backpropInfo.y2.formula}</div>
                          </div>
                          {backpropInfo.y2.components && (
                            <div className="text-xs bg-white rounded p-2">
                              {Object.entries(backpropInfo.y2.components).map(([key, val]) => (
                                <div key={key}>{key}: {val}</div>
                              ))}
                            </div>
                          )}
                          <div>
                            <div className="text-xs text-gray-600">미분값:</div>
                            <div className="font-mono font-bold text-lg">{backpropInfo.y2.derivative.toFixed(4)}</div>
                          </div>
                          <div className={`mt-2 p-2 rounded-lg ${
                            backpropInfo.y2.derivative > 0 
                              ? 'bg-green-100 border border-green-300' 
                              : 'bg-red-100 border border-red-300'
                          }`}>
                            <div className="text-xs font-medium">
                              {backpropInfo.y2.derivative > 0 ? '✓ ' : '✗ '}
                              증가 시 y₂
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
                      <strong>역전파 원리 (Sigmoid 포함):</strong> 체인 룰에 의해 각 층의 Sigmoid 미분 σ'(x) = σ(x)(1-σ(x))이 곱해집니다.
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}

      {/* XOR 게이트 테스트 패널 */}
      <div className="mt-6 bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
        <h3 className="text-2xl font-bold text-gray-800 mb-4">🎯 XOR 게이트 테스트</h3>
        <div className="mb-4 p-4 bg-blue-50 rounded-lg">
          <div className="text-sm text-gray-700 mb-2">
            <strong>목표:</strong> 가중치와 편향을 조절하여 아래 4가지 경우가 모두 정확하게 동작하도록 만들어보세요!
          </div>
          <div className="text-xs text-gray-600">
            💡 힌트: y₁은 "출력=1일 확률", y₂는 "출력=0일 확률"을 나타냅니다. 두 값을 비교하여 더 큰 값이 정답이 됩니다!
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          {[
            { x1: 0, x2: 0, expected: 0 },
            { x1: 0, x2: 1, expected: 1 },
            { x1: 1, x2: 0, expected: 1 },
            { x1: 1, x2: 1, expected: 0 }
          ].map((test, idx) => {
            // 각 테스트 케이스에 대해 출력 계산
            const tempHiddenZ = {
              z1: test.x1 * weights1.w11 + test.x2 * weights1.w12 + biases1.b1,
              z2: test.x1 * weights1.w21 + test.x2 * weights1.w22 + biases1.b2,
              z3: test.x1 * weights1.w31 + test.x2 * weights1.w32 + biases1.b3,
            };
            const tempHidden = {
              a1: sigmoid(tempHiddenZ.z1),
              a2: sigmoid(tempHiddenZ.z2),
              a3: sigmoid(tempHiddenZ.z3),
            };
            const tempOutputZ1 = tempHidden.a1 * weights2.w11 + 
                                 tempHidden.a2 * weights2.w12 + 
                                 tempHidden.a3 * weights2.w13 + 
                                 biases2.b1;
            const tempOutputZ2 = tempHidden.a1 * weights2.w21 + 
                                 tempHidden.a2 * weights2.w22 + 
                                 tempHidden.a3 * weights2.w23 + 
                                 biases2.b2;
            const result_y1 = sigmoid(tempOutputZ1);
            const result_y2 = sigmoid(tempOutputZ2);
            
            // y1(출력=1 확률)과 y2(출력=0 확률) 비교
            // expected가 1이면 y1 > y2 여야 함
            // expected가 0이면 y2 > y1 여야 함
            const predicted = result_y1 > result_y2 ? 1 : 0;
            const isCorrect = predicted === test.expected;
            
            return (
              <div 
                key={idx}
                className={`p-4 rounded-lg border-2 cursor-pointer transition-all hover:scale-105 ${
                  isCorrect 
                    ? 'bg-green-50 border-green-400' 
                    : 'bg-red-50 border-red-400'
                }`}
                onClick={() => {
                  setInputValues({ x1: test.x1, x2: test.x2 });
                }}
              >
                <div className="font-mono text-sm mb-2 font-bold">
                  입력: [{test.x1}, {test.x2}]
                </div>
                <div className="font-mono text-sm mb-2">
                  <div>기대값: <span className="font-bold text-blue-600">{test.expected}</span></div>
                  <div>y₁(=1): <span className={`font-bold ${result_y1 > result_y2 ? 'text-purple-600' : 'text-gray-500'}`}>
                    {result_y1.toFixed(3)}
                  </span></div>
                  <div>y₂(=0): <span className={`font-bold ${result_y2 > result_y1 ? 'text-purple-600' : 'text-gray-500'}`}>
                    {result_y2.toFixed(3)}
                  </span></div>
                  <div className="mt-1">예측: <span className={`font-bold ${isCorrect ? 'text-green-600' : 'text-red-600'}`}>
                    {predicted}
                  </span></div>
                </div>
                <div className={`text-xs font-bold ${
                  isCorrect ? 'text-green-700' : 'text-red-700'
                }`}>
                  {isCorrect ? '✓ 성공' : '✗ 실패'}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-4 p-3 bg-gray-50 rounded-lg text-sm text-gray-700">
          <strong>성공률:</strong> {
            [
              { x1: 0, x2: 0, expected: 0 },
              { x1: 0, x2: 1, expected: 1 },
              { x1: 1, x2: 0, expected: 1 },
              { x1: 1, x2: 1, expected: 0 }
            ].filter(test => {
              const tempHiddenZ = {
                z1: test.x1 * weights1.w11 + test.x2 * weights1.w12 + biases1.b1,
                z2: test.x1 * weights1.w21 + test.x2 * weights1.w22 + biases1.b2,
                z3: test.x1 * weights1.w31 + test.x2 * weights1.w32 + biases1.b3,
              };
              const tempHidden = {
                a1: sigmoid(tempHiddenZ.z1),
                a2: sigmoid(tempHiddenZ.z2),
                a3: sigmoid(tempHiddenZ.z3),
              };
              const tempOutputZ1 = tempHidden.a1 * weights2.w11 + 
                                   tempHidden.a2 * weights2.w12 + 
                                   tempHidden.a3 * weights2.w13 + 
                                   biases2.b1;
              const tempOutputZ2 = tempHidden.a1 * weights2.w21 + 
                                   tempHidden.a2 * weights2.w22 + 
                                   tempHidden.a3 * weights2.w23 + 
                                   biases2.b2;
              const result_y1 = sigmoid(tempOutputZ1);
              const result_y2 = sigmoid(tempOutputZ2);
              const predicted = result_y1 > result_y2 ? 1 : 0;
              return predicted === test.expected;
            }).length
          } / 4 ({
            Math.round([
              { x1: 0, x2: 0, expected: 0 },
              { x1: 0, x2: 1, expected: 1 },
              { x1: 1, x2: 0, expected: 1 },
              { x1: 1, x2: 1, expected: 0 }
            ].filter(test => {
              const tempHiddenZ = {
                z1: test.x1 * weights1.w11 + test.x2 * weights1.w12 + biases1.b1,
                z2: test.x1 * weights1.w21 + test.x2 * weights1.w22 + biases1.b2,
                z3: test.x1 * weights1.w31 + test.x2 * weights1.w32 + biases1.b3,
              };
              const tempHidden = {
                a1: sigmoid(tempHiddenZ.z1),
                a2: sigmoid(tempHiddenZ.z2),
                a3: sigmoid(tempHiddenZ.z3),
              };
              const tempOutputZ1 = tempHidden.a1 * weights2.w11 + 
                                   tempHidden.a2 * weights2.w12 + 
                                   tempHidden.a3 * weights2.w13 + 
                                   biases2.b1;
              const tempOutputZ2 = tempHidden.a1 * weights2.w21 + 
                                   tempHidden.a2 * weights2.w22 + 
                                   tempHidden.a3 * weights2.w23 + 
                                   biases2.b2;
              const result_y1 = sigmoid(tempOutputZ1);
              const result_y2 = sigmoid(tempOutputZ2);
              const predicted = result_y1 > result_y2 ? 1 : 0;
              return predicted === test.expected;
            }).length * 25)
          }%)
        </div>
      </div>
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
