import pandas as pd


def methods_diff():
    res_cols = ['method_id', 'committer_date', 'commit_hash', 'run', 'class_name', 'method_name', 'method_started_at', 'method_ended_at', 'caller_id',
                'own_duration', 'cumulative_duration', 'AVG(active)', 'AVG(available)', 'AVG(buffers)', 'AVG(cached) ', 'AVG(child_major_faults)', 'AVG(child_minor_faults)', 'AVG(commit_limit)', 'AVG(committed_as)',
'AVG(cpu_percent)', 'AVG(data)', 'AVG(dirty)', 'AVG(free)', 'AVG(high_free)', 'AVG(high_total)', 'AVG(huge_pages_total)', 'AVG(huge_pages_free)',
'AVG(hwm)', 'AVG(inactive)', 'AVG(laundry)', 'AVG(load1)', 'AVG(load5)', 'AVG(load15)', 'AVG(locked)', 'AVG(low_free)', 'AVG(low_total)', 'AVG(major_faults)', 'AVG(mapped)', 'AVG(mem_percent)',
'AVG(minor_faults)', 'AVG(page_tables)', 'AVG(pg_fault)', 'AVG(pg_in)', 'AVG(pg_maj_faults)', 'AVG(pg_out)', 'AVG(read_bytes)', 'AVG(read_count)', 'AVG(rss)', 'AVG(shared)', 'AVG(sin)', 'AVG(slab)',
'AVG(sout)', 'AVG(sreclaimable)', 'AVG(stack)', 'AVG(sunreclaim)', 'AVG(swap)', 'AVG(swap_cached)', 'AVG(swap_free)', 'AVG(swap_total)', 'AVG(swap_used)', 'AVG(swap_used_percent) ',
'AVG(total)', 'AVG(used)', 'AVG(used_percent)', 'AVG(vm_s)', 'AVG(vmalloc_chunk)', 'AVG(vmalloc_total)', 'AVG(vmalloc_used)', 'AVG(wired)', 'AVG(write_back)', 'AVG(write_back_tmp)',
'AVG(write_bytes)', 'AVG(write_count)']
    df = pd.read_csv('data/bcel-res.csv', sep=';', names=res_cols, index_col=False)
    df = df.groupby(['class_name', 'method_name', 'commit_hash', 'committer_date'])['own_duration'].agg(['count', 'mean']) \
        .reset_index()

    # print(df.head())
    df['diff'] = df.groupby(['class_name', 'method_name'])['mean'].diff()

    df['pct_ch'] = df.groupby(['class_name', 'method_name'])['mean'].pct_change()
    df.sort_values('diff', ascending=False).to_csv('results/res_diff.csv', index=False)

