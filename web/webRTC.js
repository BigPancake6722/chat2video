// 全局变量
let pc = null;
let audioContext = null;
let scriptProcessor = null;
let wavChunks = [];
const PCM_SAMPLE_RATE = 32000;  // 必须与发送端采样率一致
const BIT_DEPTH = 16;
const NUM_CHANNELS = 1;

function negotiate() {
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    return pc.createOffer().then((offer) => {
        return pc.setLocalDescription(offer);
    }).then(() => {
        // wait for ICE gathering to complete
        return new Promise((resolve) => {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                const checkState = () => {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
    }).then(() => {
        var offer = pc.localDescription;
        return fetch('/offer', {
            body: JSON.stringify({
                sdp: offer.sdp,
                type: offer.type,
            }),
            headers: {
                'Content-Type': 'application/json'
            },
            method: 'POST'
        });
    }).then((response) => {
        return response.json();
    }).then((answer) => {
        console.log("后端响应", answer)
        document.getElementById('sessionid').value = answer.sessionid
        console.log("执行sessionid赋值", sessionid)
        return pc.setRemoteDescription(answer);
    }).catch((e) => {
        alert(e);
    });
}

// WAV头生成函数
function createWavHeader(dataLength) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    
    // RIFF头
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(view, 8, 'WAVE');
    
    // fmt子块
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);          // PCM格式
    view.setUint16(22, NUM_CHANNELS, true);
    view.setUint32(24, PCM_SAMPLE_RATE, true);
    view.setUint32(28, PCM_SAMPLE_RATE * NUM_CHANNELS * (BIT_DEPTH / 8), true);
    view.setUint16(32, NUM_CHANNELS * (BIT_DEPTH / 8), true);
    view.setUint16(34, BIT_DEPTH, true);
    
    // data子块
    writeString(view, 36, 'data');
    view.setUint32(40, dataLength, true);
    
    return header;
}

function writeString(view, offset, str) {
    for (let i = 0; i < str.length; i++) {
        view.setUint8(offset + i, str.charCodeAt(i));
    }
}

// 32位浮点转16位整型
function float32ToInt16(floatArray) {
    const int16Array = new Int16Array(floatArray.length);
    for (let i = 0; i < floatArray.length; i++) {
        const val = Math.max(-1, Math.min(1, floatArray[i]));
        int16Array[i] = val < 0 ? val * 0x8000 : val * 0x7FFF;
    }
    return int16Array;
}

// 更新WAV并播放
function updateAudioPlayback() {
    if (wavChunks.length < 2) return;

    // 计算实际数据长度（排除头部）
    const dataChunks = wavChunks.slice(1);
    const totalBytes = dataChunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
    
    // 生成新头部
    const newHeader = createWavHeader(totalBytes);
    const tempData = [newHeader, ...dataChunks];
    
    // 生成Blob
    const wavBlob = new Blob(tempData, { type: 'audio/wav' });
    const audioElement = document.getElementById('audio');
    
    // 释放旧URL
    if (audioElement.src) URL.revokeObjectURL(audioElement.src);
    
    // 更新播放源
    audioElement.src = URL.createObjectURL(wavBlob);
    audioElement.play().catch(e => console.error('播放需要用户交互:', e));
    
    // 保留最近5秒数据（32000Hz * 5s = 160000 samples）
    const maxChunks = Math.ceil((5 * PCM_SAMPLE_RATE) / 4096);
    wavChunks = [newHeader, ...dataChunks.slice(-maxChunks)];
}

// 启动WebRTC连接
function start() {
    const config = {
        sdpSemantics: 'unified-plan',
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
    };

    // 初始化音频系统
    audioContext = new AudioContext({ sampleRate: PCM_SAMPLE_RATE });
    wavChunks = [createWavHeader(0)];  // 初始化空头部

    pc = new RTCPeerConnection(config);

    // 媒体流处理
    pc.ontrack = (event) => {
        if (event.track.kind === 'video') {
            document.getElementById('video').srcObject = event.streams[0];
        } else {
            const mediaStream = event.streams[0];
            
            // 创建音频处理管道
            const sourceNode = audioContext.createMediaStreamSource(mediaStream);
            scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

            // 实时音频处理
            scriptProcessor.onaudioprocess = (e) => {
                const floatData = e.inputBuffer.getChannelData(0);
                const int16Data = float32ToInt16(floatData);
                wavChunks.push(int16Data);

                // 每2秒更新一次播放
                if (audioContext.currentTime % 2 < 0.1) {
                    updateAudioPlayback();
                }
            };

            // 连接处理节点（静音输出）
            sourceNode.connect(scriptProcessor);
            scriptProcessor.connect(audioContext.destination);
        }
    };

    // 启动连接
    document.getElementById('start').disabled = true;
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

// 停止连接
function stop() {
    // 清理音频资源
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (audioContext) {
        audioContext.close().then(() => {
            audioContext = null;
            console.log('AudioContext已释放');
        });
    }
    
    // 关闭PeerConnection
    if (pc) {
        pc.close();
        pc = null;
    }
    
    // 重置UI
    document.getElementById('start').disabled = false;
    document.getElementById('stop').style.display = 'none';
}

// 其他辅助函数保持原有不变（negotiate、页面事件等）