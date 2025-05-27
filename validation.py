import re
from collections import defaultdict

def analyze_h264_stream(h264_bytes: bytes):
    """
    分析H.264流并验证其完整性，打印SPS/PPS头信息
    
    参数:
        h264_bytes: H.264原始流数据(Annex B格式)
    
    返回:
        dict: 包含分析结果的字典
    """
    # H.264 NAL单元起始码
    START_CODE = b'\x00\x00\x00\x01'
    START_CODE_LEN = len(START_CODE)
    
    # NAL单元类型定义
    NAL_UNIT_TYPES = {
        1: "非IDR帧",
        5: "IDR帧",
        6: "SEI",
        7: "SPS",
        8: "PPS",
        9: "AUD",
    }
    
    # 初始化统计信息
    stats = defaultdict(int)
    sps_pps_found = False
    sps_data = None
    pps_data = None
    
    # 查找所有NAL单元
    positions = [m.start() for m in re.finditer(START_CODE, h264_bytes)]
    
    if not positions:
        print("错误: 未找到有效的H.264起始码(00 00 00 01)")
        return None
    
    print(f"发现 {len(positions)} 个NAL单元")
    
    # 分析每个NAL单元
    for i, pos in enumerate(positions):
        nal_start = pos + START_CODE_LEN
        if i < len(positions) - 1:
            nal_end = positions[i+1]
        else:
            nal_end = len(h264_bytes)
        
        nal_data = h264_bytes[nal_start:nal_end]
        if not nal_data:
            continue
            
        # 解析NAL头
        nal_header = nal_data[0]
        nal_type = nal_header & 0x1F
        nal_ref_idc = (nal_header >> 5) & 0x3
        
        stats[nal_type] += 1
        
        # 打印NAL单元信息
        type_name = NAL_UNIT_TYPES.get(nal_type, f"未知类型({nal_type})")
        # print(f"\nNAL单元 #{i+1}:")
        # print(f"  类型: {type_name}")
        # print(f"  参考级别: {nal_ref_idc}")
        # print(f"  大小: {len(nal_data)} 字节")
        
        # 特殊处理SPS/PPS
        if nal_type == 7:  # SPS
            sps_data = nal_data
            print("  SPS内容:")
            print_hex(nal_data[:32])  # 只打印前32字节
            sps_pps_found = True
            
        elif nal_type == 8:  # PPS
            pps_data = nal_data
            print("  PPS内容:")
            print_hex(nal_data[:16])  # 只打印前16字节
            sps_pps_found = True
            
    # 验证流完整性
    print("\n完整性验证:")
    if 7 in stats and 8 in stats:
        print("✓ 包含SPS和PPS头")
    else:
        print("✗ 缺少SPS或PPS头 - 流可能无法正确解码")
    
    if 1 in stats or 5 in stats:
        print("✓ 包含视频帧数据")
    else:
        print("✗ 未找到视频帧数据")
    
    # print("\nNAL单元统计:")
    # for nal_type, count in sorted(stats.items()):
    #     type_name = NAL_UNIT_TYPES.get(nal_type, f"未知类型({nal_type})")
    #     print(f"  {type_name}: {count}个")
    
    return {
        'sps': sps_data,
        'pps': pps_data,
        'stats': dict(stats),
        'valid': (7 in stats and 8 in stats and (1 in stats or 5 in stats))
    }

def print_hex(data, bytes_per_line=8):
    """以十六进制格式打印数据"""
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i:i+bytes_per_line]
        hex_str = ' '.join(f"{b:02X}" for b in chunk)
        # print(f"    {hex_str}")

# 使用示例
def verify_h264_stream(h264_stream_bytes):
    """验证H.264流并打印信息"""
    print("="*60)
    print("H.264流分析报告")
    print("="*60)
    
    result = analyze_h264_stream(h264_stream_bytes)
    
    print("\n" + "="*60)
    if result and result['valid']:
        print("✓ H.264流验证通过")
    else:
        print("✗ H.264流存在问题")
    
    return result