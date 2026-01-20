"""
Generate methodology diagrams for the Music-Conditioned Dance Generator
Creates visual representations of the system architecture and pipeline
"""

from graphviz import Digraph


def create_methodology_diagram():
    """Create the main methodology/architecture diagram"""
    
    dot = Digraph('MusicConditionedDance_Methodology', format='png')
    dot.attr(rankdir='TB', splines='spline', nodesep='0.6', ranksep='0.8')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')


def create_methodology_diagram_4x3():
    """Create the main methodology/architecture diagram in landscape orientation"""
    
    dot = Digraph('MusicConditionedDance_Methodology_4x3', format='png')
    # Landscape orientation (left to right flow)
    dot.attr(rankdir='LR', splines='spline', nodesep='0.4', ranksep='0.4')
    dot.attr('node', fontname='Helvetica', fontsize='11')
    dot.attr('edge', fontname='Helvetica', fontsize='9')
    
    # Define colors
    input_color = '#E8F5E9'  # Light green
    audio_color = '#E3F2FD'  # Light blue
    dance_color = '#FFF3E0'  # Light orange
    latent_color = '#F3E5F5'  # Light purple
    output_color = '#FFEBEE'  # Light red
    ga_color = '#FFF8E1'     # Light yellow
    
    # ==========================================================================
    # INPUT LAYER (left side)
    # ==========================================================================
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input Data', style='rounded,filled', fillcolor=input_color, color='#4CAF50')
        c.node('dance_data', 'AIST++ Dance\nSequences\n(keypoints3d/*.pkl)', 
               shape='folder', style='filled', fillcolor='white')
        c.node('music_data', 'AIST++ Music\nTracks\n(AISTmusic/*.mp3)', 
               shape='folder', style='filled', fillcolor='white')
    
    # ==========================================================================
    # PREPROCESSING
    # ==========================================================================
    with dot.subgraph(name='cluster_preprocess') as c:
        c.attr(label='Preprocessing', style='rounded,filled', fillcolor='#ECEFF1', color='#607D8B')
        c.node('pair_data', 'Pair Dance-Music\n(match mXX# IDs)', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('extract_audio', 'Extract Audio Features\n• Mel Spectrogram (80 bands)\n• Beat Detection\n• Onset Strength', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('normalize', 'Normalize & Segment\n• 60 frames/sequence\n• Mean/Std normalization', 
               shape='box', style='rounded,filled', fillcolor='white')
    
    # ==========================================================================
    # NEURAL NETWORK - VAE
    # ==========================================================================
    with dot.subgraph(name='cluster_nn') as c:
        c.attr(label='Music-Conditioned VAE (Training)', style='rounded,filled', fillcolor='#F5F5F5', color='#9E9E9E')
        
        # Audio branch
        c.node('audio_input', 'Audio Features\n(mel + onset + beat)', 
               shape='parallelogram', style='filled', fillcolor=audio_color)
        c.node('audio_encoder', 'Audio Encoder\n(CNN + BiLSTM)', 
               shape='box3d', style='filled', fillcolor=audio_color)
        c.node('audio_embed', 'Audio Embeddings\n(per-frame + global)', 
               shape='ellipse', style='filled', fillcolor=audio_color)
        
        # Dance branch
        c.node('dance_input', 'Dance Sequence\n(60 × 51)', 
               shape='parallelogram', style='filled', fillcolor=dance_color)
        c.node('dance_encoder', 'Dance Encoder\n(BiLSTM)', 
               shape='box3d', style='filled', fillcolor=dance_color)
        c.node('mu_logvar', 'μ, log σ²', 
               shape='ellipse', style='filled', fillcolor=latent_color)
        c.node('reparam', 'Reparameterization\nz = μ + ε·σ', 
               shape='diamond', style='filled', fillcolor=latent_color)
        c.node('latent_z', 'Latent z\n(128-dim)', 
               shape='ellipse', style='filled', fillcolor=latent_color)
        
        # Decoder
        c.node('concat', 'Concatenate\n[z, audio_global]', 
               shape='invtriangle', style='filled', fillcolor='#DCEDC8')
        c.node('decoder', 'Conditional Decoder\n(MLP + Frame Refiner)', 
               shape='box3d', style='filled', fillcolor='#DCEDC8')
        c.node('recon', 'Reconstructed\nDance', 
               shape='parallelogram', style='filled', fillcolor=output_color)
        
        # Loss
        c.node('loss', 'VAE Loss\nL = Recon + β·KL', 
               shape='octagon', style='filled', fillcolor='#FFCDD2')
    
    # ==========================================================================
    # GENETIC ALGORITHM
    # ==========================================================================
    with dot.subgraph(name='cluster_ga') as c:
        c.attr(label='Genetic Algorithm (Generation)', style='rounded,filled', fillcolor=ga_color, color='#FFC107')
        
        c.node('init_pop', 'Initialize Population\n(50 genomes from\nreal dance latents)', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('decode_ga', 'Decode Genomes\n(Latent → Dance)', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('fitness', 'Evaluate Fitness\n• Motion Quality\n• Beat Alignment\n• Onset Sync', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('selection', 'Tournament\nSelection', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('crossover', 'Crossover\n(swap latent vectors)', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('mutation', 'Mutation\n(interpolate toward\nreal latents)', 
               shape='box', style='rounded,filled', fillcolor='white')
        c.node('check_gen', 'Generations\n< 100?', 
               shape='diamond', style='filled', fillcolor='white')
        c.node('best_genome', 'Best Genome', 
               shape='doubleoctagon', style='filled', fillcolor='#C8E6C9')
    
    # ==========================================================================
    # OUTPUT
    # ==========================================================================
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Output', style='rounded,filled', fillcolor=output_color, color='#F44336')
        c.node('final_dance', 'Generated Dance\n(music-synchronized)', 
               shape='note', style='filled', fillcolor='white')
        c.node('visualize', '3D Visualization\n& Animation Export', 
               shape='box', style='rounded,filled', fillcolor='white')
    
    # ==========================================================================
    # EDGES - Data Flow
    # ==========================================================================
    
    # Input to preprocessing
    dot.edge('dance_data', 'pair_data')
    dot.edge('music_data', 'pair_data')
    dot.edge('pair_data', 'extract_audio')
    dot.edge('pair_data', 'normalize')
    dot.edge('extract_audio', 'normalize')
    
    # Preprocessing to NN
    dot.edge('normalize', 'audio_input', label='audio')
    dot.edge('normalize', 'dance_input', label='dance')
    
    # Audio branch
    dot.edge('audio_input', 'audio_encoder')
    dot.edge('audio_encoder', 'audio_embed')
    
    # Dance encoder branch
    dot.edge('dance_input', 'dance_encoder')
    dot.edge('dance_encoder', 'mu_logvar')
    dot.edge('mu_logvar', 'reparam')
    dot.edge('reparam', 'latent_z')
    
    # Decoder
    dot.edge('latent_z', 'concat')
    dot.edge('audio_embed', 'concat', label='global')
    dot.edge('concat', 'decoder')
    dot.edge('audio_embed', 'decoder', label='per-frame', style='dashed')
    dot.edge('decoder', 'recon')
    
    # Loss
    dot.edge('recon', 'loss')
    dot.edge('dance_input', 'loss', style='dashed', label='target')
    dot.edge('mu_logvar', 'loss', style='dashed')
    
    # Trained model to GA
    dot.edge('loss', 'init_pop', label='trained\nmodel', style='bold', color='#4CAF50')
    
    # GA loop
    dot.edge('init_pop', 'decode_ga')
    dot.edge('decode_ga', 'fitness')
    dot.edge('fitness', 'selection')
    dot.edge('selection', 'crossover')
    dot.edge('crossover', 'mutation')
    dot.edge('mutation', 'check_gen')
    dot.edge('check_gen', 'decode_ga', label='Yes')
    dot.edge('check_gen', 'best_genome', label='No')
    
    # Music input to GA
    dot.edge('music_data', 'fitness', style='dashed', label='target\nmusic', color='#2196F3')
    
    # GA to output
    dot.edge('best_genome', 'final_dance')
    dot.edge('final_dance', 'visualize')
    
    # Render
    output_path = dot.render('methodology_diagram_4x3', cleanup=True)
    print(f"4:3 Methodology diagram saved to: {output_path}")
    
    return dot



def create_simplified_pipeline():
    """Create a simplified high-level pipeline diagram"""
    
    dot = Digraph('Pipeline_Simple', format='png')
    dot.attr(rankdir='LR', splines='spline', nodesep='1.0', ranksep='1.5')
    dot.attr('node', fontname='Helvetica', fontsize='12', shape='box', style='rounded,filled')
    
    # Nodes
    dot.node('data', 'AIST++\nDataset', fillcolor='#E8F5E9')
    dot.node('preprocess', 'Preprocess\n& Pair', fillcolor='#ECEFF1')
    dot.node('train', 'Train\nMusic-VAE', fillcolor='#E3F2FD')
    dot.node('evolve', 'GA\nEvolution', fillcolor='#FFF8E1')
    dot.node('output', 'Novel\nDance', fillcolor='#FFEBEE')
    
    # Music input
    dot.node('music', 'Target\nMusic', shape='parallelogram', fillcolor='#B3E5FC')
    
    # Edges
    dot.edge('data', 'preprocess')
    dot.edge('preprocess', 'train')
    dot.edge('train', 'evolve', label='trained\nmodel')
    dot.edge('music', 'evolve', label='conditioning')
    dot.edge('evolve', 'output')
    
    output_path = dot.render('pipeline_simple', cleanup=True)
    print(f"Simple pipeline saved to: {output_path}")
    
    return dot


def create_fitness_breakdown():
    """Create a diagram showing fitness function components"""
    
    dot = Digraph('Fitness_Breakdown', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    
    # Main fitness node
    dot.node('fitness', 'Fitness Score', shape='doubleoctagon', 
             style='filled', fillcolor='#FFF9C4', fontsize='14')
    
    # Motion quality branch
    with dot.subgraph(name='cluster_motion') as c:
        c.attr(label='Motion Quality', style='rounded,filled', fillcolor='#E8F5E9')
        c.node('smooth', 'Smoothness\n(velocity variance)')
        c.node('accel', 'Acceleration\nConsistency')
        c.node('variety', 'Movement\nVariety')
    
    # Physical plausibility branch
    with dot.subgraph(name='cluster_physical') as c:
        c.attr(label='Physical Plausibility', style='rounded,filled', fillcolor='#FFF3E0')
        c.node('floor', 'Floor\nContact')
        c.node('upright', 'Upright\nPosture')
        c.node('bones', 'Bone Length\nConsistency')
    
    # Music sync branch
    with dot.subgraph(name='cluster_music') as c:
        c.attr(label='Music Synchronization', style='rounded,filled', fillcolor='#E3F2FD')
        c.node('beat', 'Beat\nAlignment')
        c.node('onset', 'Onset\nCorrelation')
    
    # Regularization
    dot.node('latent', 'Latent Space\nRegularization', 
             shape='box', style='rounded,filled', fillcolor='#F3E5F5')
    
    # Edges
    for node in ['smooth', 'accel', 'variety']:
        dot.edge(node, 'fitness')
    for node in ['floor', 'upright', 'bones']:
        dot.edge(node, 'fitness')
    for node in ['beat', 'onset']:
        dot.edge(node, 'fitness')
    dot.edge('latent', 'fitness')
    
    output_path = dot.render('fitness_breakdown', cleanup=True)
    print(f"Fitness breakdown saved to: {output_path}")
    
    return dot


def create_architecture_comparison():
    """Create a side-by-side comparison of original vs music-conditioned"""
    
    dot = Digraph('Architecture_Comparison', format='png')
    dot.attr(rankdir='TB', nodesep='0.8', ranksep='0.6')
    dot.attr('node', fontname='Helvetica', fontsize='10')
    
    # Original (unconditional)
    with dot.subgraph(name='cluster_original') as c:
        c.attr(label='Original VAE (Unconditional)', style='rounded,filled', 
               fillcolor='#FFEBEE', color='#F44336', fontsize='12')
        c.node('o_dance', 'Dance\nSequence', shape='parallelogram', style='filled', fillcolor='white')
        c.node('o_enc', 'Dance\nEncoder', shape='box3d', style='filled', fillcolor='#FFCDD2')
        c.node('o_z', 'z', shape='circle', style='filled', fillcolor='#F3E5F5')
        c.node('o_dec', 'Decoder', shape='box3d', style='filled', fillcolor='#C8E6C9')
        c.node('o_out', 'Output\nDance', shape='parallelogram', style='filled', fillcolor='white')
        
        c.edge('o_dance', 'o_enc')
        c.edge('o_enc', 'o_z')
        c.edge('o_z', 'o_dec')
        c.edge('o_dec', 'o_out')
    
    # New (music-conditioned)
    with dot.subgraph(name='cluster_new') as c:
        c.attr(label='Music-Conditioned VAE', style='rounded,filled', 
               fillcolor='#E3F2FD', color='#2196F3', fontsize='12')
        c.node('n_dance', 'Dance\nSequence', shape='parallelogram', style='filled', fillcolor='white')
        c.node('n_music', 'Music\nFeatures', shape='parallelogram', style='filled', fillcolor='#BBDEFB')
        c.node('n_enc', 'Dance\nEncoder', shape='box3d', style='filled', fillcolor='#FFCDD2')
        c.node('n_audio', 'Audio\nEncoder', shape='box3d', style='filled', fillcolor='#90CAF9')
        c.node('n_z', 'z', shape='circle', style='filled', fillcolor='#F3E5F5')
        c.node('n_embed', 'Audio\nEmbed', shape='ellipse', style='filled', fillcolor='#90CAF9')
        c.node('n_concat', '+', shape='circle', style='filled', fillcolor='#DCEDC8')
        c.node('n_dec', 'Conditional\nDecoder', shape='box3d', style='filled', fillcolor='#C8E6C9')
        c.node('n_out', 'Output\nDance', shape='parallelogram', style='filled', fillcolor='white')
        
        c.edge('n_dance', 'n_enc')
        c.edge('n_music', 'n_audio')
        c.edge('n_enc', 'n_z')
        c.edge('n_audio', 'n_embed')
        c.edge('n_z', 'n_concat')
        c.edge('n_embed', 'n_concat')
        c.edge('n_concat', 'n_dec')
        c.edge('n_embed', 'n_dec', style='dashed', label='per-frame')
        c.edge('n_dec', 'n_out')
    
    output_path = dot.render('architecture_comparison', cleanup=True)
    print(f"Architecture comparison saved to: {output_path}")
    
    return dot


if __name__ == '__main__':
    print("Generating methodology diagrams...")
    print("=" * 50)
    
    try:
        create_methodology_diagram()
        create_methodology_diagram_4x3()
        create_simplified_pipeline()
        create_fitness_breakdown()
        create_architecture_comparison()
        
        print("\n✓ All diagrams generated successfully!")
        print("  - methodology_diagram.png (full system)")
        print("  - methodology_diagram_4x3.png (4:3 aspect ratio)")
        print("  - pipeline_simple.png (high-level pipeline)")
        print("  - fitness_breakdown.png (fitness components)")
        print("  - architecture_comparison.png (original vs new)")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure graphviz is installed:")
        print("  pip install graphviz")
        print("  brew install graphviz")
