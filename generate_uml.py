"""
UML Class Diagram Generator for musicConditionedDanceGA.py
Generates a UML diagram showing the class structure and relationships
for the Music-Conditioned Dance Generation system.
"""

from graphviz import Digraph

def create_uml_diagram():
    """Create UML class diagram for the Music-Conditioned Dance GA system"""
    
    # Create a new directed graph
    dot = Digraph('MusicConditionedDanceGA_UML', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='record', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')
    
    # ==========================================================================
    # PyTorch Dataset Class
    # ==========================================================================
    dot.node('MusicDanceDataset', '''
{MusicDanceDataset|
- dance_mean: ndarray\\l
- dance_std: ndarray\\l
- dance_data: Tensor\\l
- audio_data: Tensor\\l
- has_audio: bool\\l
|
+ __init__(dance_sequences, audio_sequences)\\l
+ __len__() → int\\l
+ __getitem__(idx) → Tuple[Tensor, Tensor]\\l
+ denormalize_dance(data) → ndarray\\l
}''')

    # ==========================================================================
    # Neural Network Classes (VAE Components)
    # ==========================================================================
    
    # nn.Module (external - simplified)
    dot.node('nn_Module', '''
{nn.Module|
(PyTorch Base Class)\\l
|
+ forward()\\l
+ parameters()\\l
+ train()\\l
+ eval()\\l
}''', style='filled', fillcolor='lightgray')

    # AudioEncoder
    dot.node('AudioEncoder', '''
{AudioEncoder|
- conv: nn.Sequential\\l
- lstm: nn.LSTM\\l
- fc: nn.Linear\\l
|
+ __init__(n_mels, hidden_dim, embed_dim)\\l
+ forward(x) → Tuple[Tensor, Tensor]\\l
}''')

    # DanceEncoder
    dot.node('DanceEncoder', '''
{DanceEncoder|
- lstm: nn.LSTM\\l
- fc_mu: nn.Linear\\l
- fc_logvar: nn.Linear\\l
|
+ __init__(input_dim, hidden_dim, latent_dim)\\l
+ forward(x: Tensor) → Tuple[Tensor, Tensor]\\l
}''')

    # ConditionalDanceDecoder
    dot.node('ConditionalDanceDecoder', '''
{ConditionalDanceDecoder|
- seq_len: int\\l
- output_dim: int\\l
- decoder: nn.Sequential\\l
- frame_refiner: nn.Sequential\\l
|
+ __init__(latent_dim, audio_embed_dim, hidden_dim, output_dim)\\l
+ forward(z, audio_embed, audio_global) → Tensor\\l
}''')

    # MusicConditionedVAE
    dot.node('MusicConditionedVAE', '''
{MusicConditionedVAE|
- audio_encoder: AudioEncoder\\l
- dance_encoder: DanceEncoder\\l
- decoder: ConditionalDanceDecoder\\l
|
+ __init__()\\l
+ reparameterize(mu, logvar) → Tensor\\l
+ forward(dance, audio) → Tuple\\l
+ encode(dance) → Tensor\\l
+ decode(z, audio) → Tensor\\l
+ generate_from_music(audio, num_samples) → Tensor\\l
}''')

    # ==========================================================================
    # Genetic Algorithm Class
    # ==========================================================================
    
    dot.node('MusicConditionedGenome', '''
{MusicConditionedGenome|
- latent_vectors: List[ndarray]\\l
- audio_features: ndarray | None\\l
- fitness: float | None\\l
- decoded_frames: ndarray | None\\l
|
+ __init__(latent_vectors, audio_features)\\l
}''')

    # ==========================================================================
    # Configuration/Constants (as a note)
    # ==========================================================================
    dot.node('Config', '''
{«configuration»\\nSettings|
SEQUENCE_LENGTH = 60\\l
LATENT_DIM = 128\\l
HIDDEN_DIM = 512\\l
AUDIO_HIDDEN_DIM = 256\\l
AUDIO_EMBED_DIM = 64\\l
N_MELS = 80\\l
NUM_KEYPOINTS = 17\\l
INPUT_DIM = 51\\l
|
POPULATION_SIZE = 50\\l
NUM_GENERATIONS = 100\\l
MUTATION_RATE = 0.15\\l
CROSSOVER_RATE = 0.7\\l
}''', style='filled', fillcolor='lightyellow')

    # ==========================================================================
    # Function Groups (as utility classes)
    # ==========================================================================
    
    dot.node('AudioUtils', '''
{«utility»\\nAudio Processing|
+ extract_music_id(filename) → str\\l
+ load_audio_features(audio_path, duration) → dict\\l
+ resample_audio_to_dance_fps(features, frames) → dict\\l
+ load_music_cache(music_folder) → dict\\l
}''', style='filled', fillcolor='#E3F2FD')
    
    dot.node('DataUtils', '''
{«utility»\\nData Loading|
+ load_pkl(filepath) → ndarray\\l
+ load_paired_sequences(folder, music, seq_len, max) → Tuple\\l
+ load_all_sequences_unconditional(folder, seq_len, max) → Tuple\\l
}''', style='filled', fillcolor='lightblue')
    
    dot.node('TrainingUtils', '''
{«utility»\\nTraining|
+ train_music_vae(model, dataloader, epochs) → dict\\l
}''', style='filled', fillcolor='lightblue')
    
    dot.node('GAOperations', '''
{«utility»\\nGA Operations|
+ decode_music_genome(genome, model, dataset) → ndarray\\l
+ decode_genome_unconditional(genome, model, dataset) → ndarray\\l
+ smooth_frames(frames, window_size) → ndarray\\l
+ calculate_music_sync_fitness(genome, model, dataset) → float\\l
+ tournament_select(population) → MusicConditionedGenome\\l
+ crossover_music(parent1, parent2) → MusicConditionedGenome\\l
+ mutate_music(genome, real_latents) → MusicConditionedGenome\\l
+ get_real_dance_latents(model, dataset, num) → List\\l
+ run_music_ga_evolution(model, dataset, target_music) → MusicConditionedGenome\\l
}''', style='filled', fillcolor='lightgreen')
    
    dot.node('Visualization', '''
{«utility»\\nVisualization|
+ get_bone_color(bone_idx) → str\\l
+ visualize_dance_with_music(frames, audio, title)\\l
+ save_animation(frames, filename)\\l
+ get_next_filename(base, ext) → str\\l
}''', style='filled', fillcolor='lightsalmon')

    # ==========================================================================
    # Relationships
    # ==========================================================================
    
    # Inheritance
    dot.edge('nn_Module', 'AudioEncoder', label='inherits', arrowhead='empty', style='solid')
    dot.edge('nn_Module', 'DanceEncoder', label='inherits', arrowhead='empty', style='solid')
    dot.edge('nn_Module', 'ConditionalDanceDecoder', label='inherits', arrowhead='empty', style='solid')
    dot.edge('nn_Module', 'MusicConditionedVAE', label='inherits', arrowhead='empty', style='solid')
    
    # Composition (MusicConditionedVAE contains encoders and decoder)
    dot.edge('MusicConditionedVAE', 'AudioEncoder', label='1', arrowhead='diamond', style='solid', dir='back')
    dot.edge('MusicConditionedVAE', 'DanceEncoder', label='1', arrowhead='diamond', style='solid', dir='back')
    dot.edge('MusicConditionedVAE', 'ConditionalDanceDecoder', label='1', arrowhead='diamond', style='solid', dir='back')
    
    # Association/Dependency
    dot.edge('GAOperations', 'MusicConditionedGenome', label='creates/uses', style='dashed', arrowhead='open')
    dot.edge('GAOperations', 'MusicConditionedVAE', label='uses', style='dashed', arrowhead='open')
    dot.edge('GAOperations', 'MusicDanceDataset', label='uses', style='dashed', arrowhead='open')
    
    dot.edge('TrainingUtils', 'MusicConditionedVAE', label='trains', style='dashed', arrowhead='open')
    dot.edge('TrainingUtils', 'MusicDanceDataset', label='uses', style='dashed', arrowhead='open')
    
    dot.edge('AudioUtils', 'MusicDanceDataset', label='provides audio to', style='dashed', arrowhead='open')
    dot.edge('DataUtils', 'MusicDanceDataset', label='provides dance to', style='dashed', arrowhead='open')
    
    dot.edge('Visualization', 'MusicConditionedGenome', label='visualizes', style='dashed', arrowhead='open')
    
    # Config dependencies
    dot.edge('Config', 'MusicConditionedVAE', style='dotted', arrowhead='open')
    dot.edge('Config', 'GAOperations', style='dotted', arrowhead='open')
    dot.edge('Config', 'AudioUtils', style='dotted', arrowhead='open')
    
    # ==========================================================================
    # Layout hints using subgraphs
    # ==========================================================================
    
    with dot.subgraph(name='cluster_nn') as c:
        c.attr(label='Neural Network (Music-Conditioned VAE)', style='rounded', color='blue')
        c.node('AudioEncoder')
        c.node('DanceEncoder')
        c.node('ConditionalDanceDecoder')
        c.node('MusicConditionedVAE')
    
    with dot.subgraph(name='cluster_ga') as c:
        c.attr(label='Genetic Algorithm', style='rounded', color='green')
        c.node('MusicConditionedGenome')
        c.node('GAOperations')
    
    with dot.subgraph(name='cluster_audio') as c:
        c.attr(label='Audio Processing', style='rounded', color='#2196F3')
        c.node('AudioUtils')
    
    # Render the diagram
    output_path = dot.render('music_conditioned_uml_class_diagram', cleanup=True)
    print(f"UML diagram saved to: {output_path}")
    
    return dot


def create_sequence_diagram():
    """Create a simplified sequence/flow diagram for music-conditioned dance generation"""
    
    dot = Digraph('MusicConditionedDanceGA_Flow', format='png')
    dot.attr(rankdir='LR', splines='polyline')
    dot.attr('node', shape='box', fontname='Helvetica', fontsize='10', style='rounded')
    
    # Main flow nodes
    dot.node('start', 'Start', shape='ellipse')
    dot.node('load_dance', 'Load Dance\nSequences')
    dot.node('load_music', 'Load Music\nFeatures')
    dot.node('pair', 'Pair Dance\n& Music')
    dot.node('dataset', 'Create\nMusicDanceDataset')
    dot.node('train', 'Train Music-\nConditioned VAE\n(or load)')
    dot.node('select_music', 'Select Target\nMusic')
    dot.node('encode', 'Encode Real\nDances')
    dot.node('init_pop', 'Initialize\nPopulation')
    dot.node('fitness', 'Evaluate\nMusic-Sync Fitness')
    dot.node('evolve', 'Selection +\nCrossover +\nMutation')
    dot.node('check', 'Generations\nComplete?', shape='diamond')
    dot.node('decode', 'Decode Best\nGenome')
    dot.node('visualize', 'Visualize Dance\n+ Music')
    dot.node('end', 'End', shape='ellipse')
    
    # Edges
    dot.edge('start', 'load_dance')
    dot.edge('load_dance', 'load_music')
    dot.edge('load_music', 'pair')
    dot.edge('pair', 'dataset')
    dot.edge('dataset', 'train')
    dot.edge('train', 'select_music')
    dot.edge('select_music', 'encode')
    dot.edge('encode', 'init_pop')
    dot.edge('init_pop', 'fitness')
    dot.edge('fitness', 'evolve')
    dot.edge('evolve', 'check')
    dot.edge('check', 'fitness', label='No')
    dot.edge('check', 'decode', label='Yes')
    dot.edge('decode', 'visualize')
    dot.edge('visualize', 'end')
    
    output_path = dot.render('music_conditioned_flow_diagram', cleanup=True)
    print(f"Flow diagram saved to: {output_path}")
    
    return dot


def create_music_conditioned_flow_diagram():
    """Create a 4:3 flow diagram for the music-conditioned autoencoder"""
    
    dot = Digraph('MusicConditionedDanceGA_Flow', format='png')
    # Force 4:3 aspect ratio (e.g., 12x9 inches)
    dot.attr(rankdir='TB', splines='polyline', nodesep='0.3', ranksep='0.4')
    dot.attr(size='12,9!', ratio='fill', dpi='150')
    dot.attr('node', shape='box', fontname='Helvetica', fontsize='9', style='rounded', width='1.2', height='0.5')
    
    # Colors
    audio_color = '#E3F2FD'
    dance_color = '#FFF3E0'
    ga_color = '#FFF8E1'
    
    # Row 1
    dot.node('start', 'Start', shape='ellipse', style='filled', fillcolor='#C8E6C9')
    dot.node('load_dance', 'Load Dance', style='filled', fillcolor=dance_color)
    dot.node('load_music', 'Load Music', style='filled', fillcolor=audio_color)
    dot.node('pair', 'Pair Data', style='filled', fillcolor='#ECEFF1')
    dot.node('extract', 'Extract\nFeatures', style='filled', fillcolor=audio_color)
    
    # Row 2
    dot.node('dataset', 'Create\nDataset', style='filled', fillcolor='#F5F5F5')
    dot.node('train', 'Train VAE', style='filled', fillcolor='#DCEDC8')
    dot.node('select', 'Select\nMusic', shape='parallelogram', style='filled', fillcolor=audio_color)
    dot.node('encode', 'Encode\nLatents', style='filled', fillcolor='#F3E5F5')
    dot.node('init', 'Init Pop', style='filled', fillcolor=ga_color)
    
    # Row 3
    dot.node('decode', 'Decode', style='filled', fillcolor=ga_color)
    dot.node('fitness', 'Fitness', style='filled', fillcolor=ga_color)
    dot.node('selection', 'Select', style='filled', fillcolor=ga_color)
    dot.node('cross', 'Crossover', style='filled', fillcolor=ga_color)
    dot.node('mutate', 'Mutate', style='filled', fillcolor=ga_color)
    
    # Row 4
    dot.node('check', 'Done?', shape='diamond', style='filled', fillcolor=ga_color)
    dot.node('best', 'Best', style='filled', fillcolor='#C8E6C9')
    dot.node('decode2', 'Decode\nDance', style='filled', fillcolor='#DCEDC8')
    dot.node('viz', 'Visualize', style='filled', fillcolor='#FFCDD2')
    dot.node('end', 'End', shape='ellipse', style='filled', fillcolor='#C8E6C9')
    
    # Row constraints
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('start'); s.node('load_dance'); s.node('load_music'); s.node('pair'); s.node('extract')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('dataset'); s.node('train'); s.node('select'); s.node('encode'); s.node('init')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('decode'); s.node('fitness'); s.node('selection'); s.node('cross'); s.node('mutate')
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('check'); s.node('best'); s.node('decode2'); s.node('viz'); s.node('end')
    
    # Row 1 edges
    dot.edge('start', 'load_dance')
    dot.edge('load_dance', 'load_music')
    dot.edge('load_music', 'pair')
    dot.edge('pair', 'extract')
    
    # Row 1→2 and Row 2 edges
    dot.edge('extract', 'dataset')
    dot.edge('dataset', 'train')
    dot.edge('train', 'select')
    dot.edge('select', 'encode')
    dot.edge('encode', 'init')
    
    # Row 2→3 and Row 3 edges
    dot.edge('init', 'decode')
    dot.edge('decode', 'fitness')
    dot.edge('fitness', 'selection')
    dot.edge('selection', 'cross')
    dot.edge('cross', 'mutate')
    
    # Row 3→4 and Row 4 edges
    dot.edge('mutate', 'check')
    dot.edge('check', 'decode', label='Y', style='dashed', constraint='false')
    dot.edge('check', 'best', label='N')
    dot.edge('best', 'decode2')
    dot.edge('decode2', 'viz')
    dot.edge('viz', 'end')
    
    # Music conditioning
    dot.edge('select', 'fitness', style='dashed', color='#2196F3', constraint='false')
    
    output_path = dot.render('music_conditioned_flow_diagram', cleanup=True)
    print(f"Music-conditioned flow diagram saved to: {output_path}")
    return dot


if __name__ == '__main__':
    print("Generating UML diagrams for Music-Conditioned Dance GA...")
    print("=" * 50)
    
    try:
        class_diagram = create_uml_diagram()
        flow_diagram = create_sequence_diagram()
        music_flow = create_music_conditioned_flow_diagram()
        
        print("\n✓ Diagrams generated successfully!")
        print("  - music_conditioned_uml_class_diagram.png")
        print("  - music_conditioned_flow_diagram.png (sequence)")
        print("  - music_conditioned_flow_diagram.png (detailed)")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure graphviz is installed:")
        print("  pip install graphviz")
        print("  brew install graphviz  (macOS)")
