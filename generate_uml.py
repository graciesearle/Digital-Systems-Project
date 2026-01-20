"""
UML Class Diagram Generator for autoencoderDanceGA.py
Generates a UML diagram showing the class structure and relationships
"""

from graphviz import Digraph

def create_uml_diagram():
    """Create UML class diagram for the Autoencoder Dance GA system"""
    
    # Create a new directed graph
    dot = Digraph('AutoencoderDanceGA_UML', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.8', ranksep='1.2')
    dot.attr('node', shape='record', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')
    
    # ==========================================================================
    # PyTorch Dataset Class
    # ==========================================================================
    dot.node('DanceDataset', '''
{DanceDataset|
- mean: ndarray\\l
- std: ndarray\\l
- data: Tensor\\l
|
+ __init__(sequences: ndarray)\\l
+ __len__() → int\\l
+ __getitem__(idx: int) → Tensor\\l
+ denormalize(data) → ndarray\\l
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

    # DanceDecoder
    dot.node('DanceDecoder', '''
{DanceDecoder|
- seq_len: int\\l
- output_dim: int\\l
- decoder: nn.Sequential\\l
|
+ __init__(latent_dim, hidden_dim, output_dim)\\l
+ forward(z, target, teacher_forcing_ratio, seq_len) → Tensor\\l
}''')

    # DanceVAE
    dot.node('DanceVAE', '''
{DanceVAE|
- encoder: DanceEncoder\\l
- decoder: DanceDecoder\\l
|
+ __init__()\\l
+ reparameterize(mu, logvar) → Tensor\\l
+ forward(x, teacher_forcing_ratio) → Tuple\\l
+ encode(x) → Tensor\\l
+ decode(z) → Tensor\\l
}''')

    # ==========================================================================
    # Genetic Algorithm Class
    # ==========================================================================
    
    dot.node('LatentGenome', '''
{LatentGenome|
- latent_vectors: List[ndarray]\\l
- fitness: float | None\\l
- decoded_frames: ndarray | None\\l
|
+ __init__(latent_vectors: List[ndarray] | None)\\l
}''')

    # ==========================================================================
    # Configuration/Constants (as a note)
    # ==========================================================================
    dot.node('Config', '''
{«configuration»\\nSettings|
SEQUENCE_LENGTH = 60\\l
LATENT_DIM = 128\\l
HIDDEN_DIM = 512\\l
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
    
    dot.node('DataUtils', '''
{«utility»\\nData Loading|
+ extract_genre(filename) → str\\l
+ load_pkl(filepath) → ndarray\\l
+ load_all_sequences(folder, seq_len, max_files) → ndarray\\l
}''', style='filled', fillcolor='lightblue')
    
    dot.node('TrainingUtils', '''
{«utility»\\nTraining|
+ vae_loss(recon_x, x, mu, logvar, beta) → Tensor\\l
+ train_autoencoder(model, dataloader, epochs) → dict\\l
}''', style='filled', fillcolor='lightblue')
    
    dot.node('GAOperations', '''
{«utility»\\nGA Operations|
+ decode_genome(genome, model, dataset) → ndarray\\l
+ smooth_frames(frames, window_size) → ndarray\\l
+ calculate_fitness(genome, model, dataset) → float\\l
+ tournament_select(population) → LatentGenome\\l
+ crossover(parent1, parent2) → LatentGenome\\l
+ mutate(genome, real_latents) → LatentGenome\\l
+ interpolate_latent(genome) → LatentGenome\\l
+ get_real_dance_latents(model, dataset, num) → List\\l
+ run_ga_evolution(model, dataset) → LatentGenome\\l
}''', style='filled', fillcolor='lightgreen')
    
    dot.node('Visualization', '''
{«utility»\\nVisualization|
+ get_bone_color(bone_idx) → str\\l
+ visualize_dance(frames, title)\\l
+ save_animation(frames, filename)\\l
+ get_next_filename(base, ext) → str\\l
}''', style='filled', fillcolor='lightsalmon')

    # ==========================================================================
    # Relationships
    # ==========================================================================
    
    # Inheritance
    dot.edge('nn_Module', 'DanceEncoder', label='inherits', arrowhead='empty', style='solid')
    dot.edge('nn_Module', 'DanceDecoder', label='inherits', arrowhead='empty', style='solid')
    dot.edge('nn_Module', 'DanceVAE', label='inherits', arrowhead='empty', style='solid')
    
    # Composition (DanceVAE contains encoder and decoder)
    dot.edge('DanceVAE', 'DanceEncoder', label='1', arrowhead='diamond', style='solid', dir='back')
    dot.edge('DanceVAE', 'DanceDecoder', label='1', arrowhead='diamond', style='solid', dir='back')
    
    # Association/Dependency
    dot.edge('GAOperations', 'LatentGenome', label='creates/uses', style='dashed', arrowhead='open')
    dot.edge('GAOperations', 'DanceVAE', label='uses', style='dashed', arrowhead='open')
    dot.edge('GAOperations', 'DanceDataset', label='uses', style='dashed', arrowhead='open')
    
    dot.edge('TrainingUtils', 'DanceVAE', label='trains', style='dashed', arrowhead='open')
    dot.edge('TrainingUtils', 'DanceDataset', label='uses', style='dashed', arrowhead='open')
    
    dot.edge('DataUtils', 'DanceDataset', label='provides data to', style='dashed', arrowhead='open')
    
    dot.edge('Visualization', 'LatentGenome', label='visualizes', style='dashed', arrowhead='open')
    
    # Config dependencies
    dot.edge('Config', 'DanceVAE', style='dotted', arrowhead='open')
    dot.edge('Config', 'GAOperations', style='dotted', arrowhead='open')
    
    # ==========================================================================
    # Layout hints using subgraphs
    # ==========================================================================
    
    with dot.subgraph(name='cluster_nn') as c:
        c.attr(label='Neural Network (VAE)', style='rounded', color='blue')
        c.node('DanceEncoder')
        c.node('DanceDecoder')
        c.node('DanceVAE')
    
    with dot.subgraph(name='cluster_ga') as c:
        c.attr(label='Genetic Algorithm', style='rounded', color='green')
        c.node('LatentGenome')
        c.node('GAOperations')
    
    # Render the diagram
    output_path = dot.render('uml_class_diagram', cleanup=True)
    print(f"UML diagram saved to: {output_path}")
    
    return dot


def create_sequence_diagram():
    """Create a simplified sequence/flow diagram"""
    
    dot = Digraph('AutoencoderDanceGA_Flow', format='png')
    dot.attr(rankdir='LR', splines='polyline')
    dot.attr('node', shape='box', fontname='Helvetica', fontsize='10', style='rounded')
    
    # Main flow nodes
    dot.node('start', 'Start', shape='ellipse')
    dot.node('load', 'Load Dance\nSequences')
    dot.node('dataset', 'Create\nDanceDataset')
    dot.node('train', 'Train VAE\n(or load)')
    dot.node('encode', 'Encode Real\nDances')
    dot.node('init_pop', 'Initialize\nPopulation')
    dot.node('fitness', 'Evaluate\nFitness')
    dot.node('evolve', 'Selection +\nCrossover +\nMutation')
    dot.node('check', 'Generations\nComplete?', shape='diamond')
    dot.node('decode', 'Decode Best\nGenome')
    dot.node('visualize', 'Visualize\nDance')
    dot.node('end', 'End', shape='ellipse')
    
    # Edges
    dot.edge('start', 'load')
    dot.edge('load', 'dataset')
    dot.edge('dataset', 'train')
    dot.edge('train', 'encode')
    dot.edge('encode', 'init_pop')
    dot.edge('init_pop', 'fitness')
    dot.edge('fitness', 'evolve')
    dot.edge('evolve', 'check')
    dot.edge('check', 'fitness', label='No')
    dot.edge('check', 'decode', label='Yes')
    dot.edge('decode', 'visualize')
    dot.edge('visualize', 'end')
    
    output_path = dot.render('uml_flow_diagram', cleanup=True)
    print(f"Flow diagram saved to: {output_path}")
    
    return dot


if __name__ == '__main__':
    print("Generating UML diagrams for autoencoderDanceGA.py...")
    print("=" * 50)
    
    try:
        # Generate class diagram
        class_diagram = create_uml_diagram()
        
        # Generate flow diagram
        flow_diagram = create_sequence_diagram()
        
        print("\n✓ Diagrams generated successfully!")
        print("  - uml_class_diagram.png")
        print("  - uml_flow_diagram.png")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure graphviz is installed:")
        print("  pip install graphviz")
        print("  brew install graphviz  (macOS)")
