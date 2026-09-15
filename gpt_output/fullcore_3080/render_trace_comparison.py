"""Standalone interactive timeline and common-boundary CSV for tagged captures."""
import argparse, csv, ctypes as c, html, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
rsvg=c.CDLL('librsvg-2.so.2');cairo=c.CDLL('libcairo.so.2');gobject=c.CDLL('libgobject-2.0.so.0')
rsvg.rsvg_handle_new_from_data.argtypes=[c.c_char_p,c.c_size_t,c.c_void_p];rsvg.rsvg_handle_new_from_data.restype=c.c_void_p
cairo.cairo_image_surface_create.argtypes=[c.c_int,c.c_int,c.c_int];cairo.cairo_image_surface_create.restype=c.c_void_p
cairo.cairo_create.argtypes=[c.c_void_p];cairo.cairo_create.restype=c.c_void_p
rsvg.rsvg_handle_render_cairo.argtypes=[c.c_void_p,c.c_void_p];rsvg.rsvg_handle_render_cairo.restype=c.c_int
cairo.cairo_surface_write_to_png.argtypes=[c.c_void_p,c.c_char_p];cairo.cairo_surface_write_to_png.restype=c.c_int
cairo.cairo_destroy.argtypes=[c.c_void_p];cairo.cairo_surface_destroy.argtypes=[c.c_void_p];gobject.g_object_unref.argtypes=[c.c_void_p]
rows=json.loads((a.root/'summary.json').read_text());panels=[];buttons=[]
for row in rows:
    directory=a.root/f"{row['order']:02d}_cpu{row['cpu_cores']}"
    events=json.loads((directory/'timers_host_events.json').read_text())
    for stage in ('cpu_stage','expert_forward_cpu','expert_forward_cuda','gpu_hit_submit','background_wait','moe_layer','weight_copy_submit','eviction_select','expert_acquire'):
        row[stage+'_timers_ms_per_token']=sum((e['end_ns']-e['start_ns'])/1e6 for e in events if e['stage']==stage)/5
    label=f"CPU{row['cpu_cores']} / {row['policy']}"
    if (directory/'overlap_metrics.json').exists():
        metrics=json.loads((directory/'overlap_metrics.json').read_text())
        for k,v in metrics['resource_union_ms'].items():row['trace_'+k+'_ms_per_token']=v/5
        buttons.append(f'<button onclick="show({row["order"]})">{html.escape(label)}</button>')
        for stem in ('token_timeline','layer13_timeline','weight_transfer_timeline','weight_transfer_token_timeline'):
            source=directory/(stem+'.svg')
            if not source.exists():continue
            b=source.read_bytes();handle=rsvg.rsvg_handle_new_from_data(b,len(b),None)
            surface=cairo.cairo_image_surface_create(0,1240,440);ctx=cairo.cairo_create(surface)
            assert rsvg.rsvg_handle_render_cairo(handle,ctx)
            assert cairo.cairo_surface_write_to_png(surface,str(source.with_suffix('.png')).encode())==0
            cairo.cairo_destroy(ctx);cairo.cairo_surface_destroy(surface);gobject.g_object_unref(handle)
            panels.append(f'<section data-order="{row["order"]}"><h2>{html.escape(label)}</h2>{b.decode()}</section>')
keys=list(dict.fromkeys(k for row in rows for k in row))
with (a.root/'comparison.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SMoE overlap 对照</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#203448}svg{width:100%;height:auto}section{border:1px solid #d5dfe8;margin:24px 0;padding:12px}button{padding:8px;margin:5px}p{line-height:1.7}h2{font-size:18px}</style>
<h1>CPU 专家 / GPU 专家 / PCIe 重叠对照</h1>
<p>固定选择第三个捕获token及第13层；各运行实际路由工作量可能不同。绿色为CPU专家调用墙钟，蓝色为真实GPU执行，橙紫为PCIe传输。GPU总执行已包含专家，不能相加。native批量模式的CPU专家范围来自C++ ATen数学，原路径包括Python模块调用；共同边界CPU stage另见CSV。Profiler有明显开销，不能把trace窗口时间当验收结果，也不能单凭overlap百分比判定加速。</p>
<button onclick="show(-1)">显示全部</button>'''+''.join(buttons)+''.join(panels)+'''
<script>function show(k){document.querySelectorAll('section').forEach(x=>x.style.display=k<0||+x.dataset.order===k?'block':'none')}</script></html>'''
(a.root/'timeline.html').write_text(page)
