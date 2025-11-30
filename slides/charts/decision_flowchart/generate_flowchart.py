"""
Generate Strategy 2 Decision Flowchart using Graphviz.
Output: decision_flowchart.pdf
"""

from graphviz import Digraph

def create_flowchart():
    dot = Digraph(comment='Strategy 2 Decision Flowchart')
    dot.attr(rankdir='TB', size='8,10')
    dot.attr('node', shape='box', style='rounded,filled', fontsize='10')

    # Start node
    dot.node('start', 'Day t\nRemaining USD > 0?', fillcolor='lightblue')

    # Initial period check
    dot.node('initial', 't < 10?', shape='diamond', fillcolor='lightyellow')

    # Initial period action
    dot.node('const', 'Execute:\nbase_daily\n(constant)', fillcolor='lightgreen')

    # Price vs benchmark
    dot.node('compare', 'Price < Benchmark?', shape='diamond', fillcolor='lightyellow')

    # Favorable branch
    dot.node('past_min', 't >= min_duration?', shape='diamond', fillcolor='lightyellow')

    # Past min duration - ASAP
    dot.node('asap', 'Execute:\nmin(5x base, remaining)\n(finish ASAP)', fillcolor='lightgreen')

    # Before min duration - speedup
    dot.node('speedup', 'Execute:\nmin(remaining/days_to_min,\n5x base, remaining)', fillcolor='lightgreen')

    # Unfavorable branch
    dot.node('extra_slow_check', 'remaining < 5x base\nAND days_to_max > 5?', shape='diamond', fillcolor='lightyellow')

    # Extra slow
    dot.node('extra_slow', 'Execute:\n0.1x base\n(extra slow)', fillcolor='lightsalmon')

    # Normal slow
    dot.node('slow', 'Execute:\nremaining/days_to_max\n(slow down)', fillcolor='lightsalmon')

    # End node
    dot.node('next', 'Next Day\n(t = t + 1)', fillcolor='lightgray')
    dot.node('done', 'DONE\n(execution complete)', fillcolor='lightblue')

    # Edges
    dot.edge('start', 'initial', label='Yes')
    dot.edge('start', 'done', label='No')

    dot.edge('initial', 'const', label='Yes\n(initial period)')
    dot.edge('initial', 'compare', label='No\n(adaptive period)')

    dot.edge('const', 'next')

    dot.edge('compare', 'past_min', label='Yes\n(favorable)')
    dot.edge('compare', 'extra_slow_check', label='No\n(unfavorable)')

    dot.edge('past_min', 'asap', label='Yes')
    dot.edge('past_min', 'speedup', label='No')

    dot.edge('asap', 'next')
    dot.edge('speedup', 'next')

    dot.edge('extra_slow_check', 'extra_slow', label='Yes')
    dot.edge('extra_slow_check', 'slow', label='No')

    dot.edge('extra_slow', 'next')
    dot.edge('slow', 'next')

    dot.edge('next', 'start')

    return dot

if __name__ == '__main__':
    import os

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create flowchart
    dot = create_flowchart()

    # Render to PDF
    output_path = os.path.join(script_dir, 'decision_flowchart')
    dot.render(output_path, format='pdf', cleanup=True)

    print(f"Generated: {output_path}.pdf")
