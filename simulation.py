import numpy as np
import cupy as cp
import random
from collections import defaultdict
import time

class GPUBlackjackSimulator:
    def __init__(self, batch_size=10000):
        self.batch_size = batch_size
        self.num_decks = 6
        self.min_bet = 25
        self.initial_bankroll = 10000
        
        # Initialize GPU arrays
        self.setup_gpu_arrays()
        self.setup_strategy_tables()
        
    def setup_gpu_arrays(self):
        """Initialize GPU arrays for batch processing"""
        # Card values for Hi-Lo counting
        self.card_values = cp.array([
            1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1  # 2-6: +1, 7-9: 0, 10-A: -1
        ], dtype=cp.int32)
        
        # Blackjack values
        self.bj_values = cp.array([
            2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11  # 2-A
        ], dtype=cp.int32)
        
        # Initialize shoe (6 decks)
        single_deck = cp.tile(cp.arange(13, dtype=cp.int32), 4)  # 4 suits
        self.full_shoe = cp.tile(single_deck, self.num_decks)
        
    def setup_strategy_tables(self):
        """Setup basic strategy tables on GPU"""
        # Hard strategy table (17x10) - dealer 2-A vs player 5-21
        self.hard_strategy = cp.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 6
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 8
            [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],  # 9
            [2, 2, 2, 2, 2, 2, 2, 2, 0, 0],  # 10
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 0],  # 11
            [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],  # 12
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 13
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 14
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 15
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # 16
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 17
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 18
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 19
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 20
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # 21
        ], dtype=cp.int32)
        
        # Soft strategy table (9x10) - dealer 2-A vs soft 13-21
        self.soft_strategy = cp.array([
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],  # A,2 (soft 13)
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],  # A,3 (soft 14)
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],  # A,4 (soft 15)
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],  # A,5 (soft 16)
            [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],  # A,6 (soft 17)
            [1, 2, 2, 2, 2, 1, 1, 0, 0, 0],  # A,7 (soft 18)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,8 (soft 19)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,9 (soft 20)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,10 (soft 21)
        ], dtype=cp.int32)
        
        # Pair splitting strategy (13x10)
        self.pair_strategy = cp.array([
            [3, 3, 3, 3, 3, 3, 0, 0, 0, 0],  # 2,2
            [3, 3, 3, 3, 3, 3, 0, 0, 0, 0],  # 3,3
            [0, 0, 0, 3, 3, 0, 0, 0, 0, 0],  # 4,4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 5,5
            [3, 3, 3, 3, 3, 0, 0, 0, 0, 0],  # 6,6
            [3, 3, 3, 3, 3, 3, 0, 0, 0, 0],  # 7,7
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # 8,8
            [3, 3, 3, 3, 3, 0, 3, 3, 0, 0],  # 9,9
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 10,10
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # J,J
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Q,Q
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # K,K
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # A,A
        ], dtype=cp.int32)
        
        # Action codes: 0=Hit, 1=Stand, 2=Double, 3=Split
        
    def shuffle_shoes(self, num_games):
        """Generate shuffled shoes for batch processing"""
        shoes = cp.zeros((num_games, len(self.full_shoe)), dtype=cp.int32)
        
        for i in range(num_games):
            # Create shuffled shoe
            shoe = cp.copy(self.full_shoe)
            # Use cupy's random shuffle
            cp.random.shuffle(shoe)
            shoes[i] = shoe
            
        return shoes
    
    def calculate_hand_value(self, cards, num_cards):
        """Calculate hand value on GPU"""
        # Convert card indices to values
        values = self.bj_values[cards]
        
        # Calculate soft total
        total = cp.sum(values, axis=1)
        aces = cp.sum(cards == 12, axis=1)  # Count aces (index 12)
        
        # Adjust for aces
        while cp.any((total > 21) & (aces > 0)):
            mask = (total > 21) & (aces > 0)
            total = cp.where(mask, total - 10, total)
            aces = cp.where(mask, aces - 1, aces)
            
        return total
    
    def get_betting_amounts(self, true_counts):
        """Calculate betting amounts based on true count"""
        bet_amounts = cp.where(true_counts < 1, 25,
                      cp.where(true_counts < 2, 100,
                      cp.where(true_counts < 3, 200,
                      cp.where(true_counts < 4, 400,
                      cp.where(true_counts < 5, 400, 500)))))
        
        # Number of hands (1 for TC<4, 2 for TC>=4)
        num_hands = cp.where(true_counts < 4, 1, 2)
        
        return bet_amounts, num_hands
    
    def play_batch_gpu(self, num_games):
        """Play multiple blackjack games in parallel on GPU"""
        # Initialize game states
        shoes = self.shuffle_shoes(num_games)
        shoe_positions = cp.zeros(num_games, dtype=cp.int32)
        running_counts = cp.zeros(num_games, dtype=cp.int32)
        bankrolls = cp.full(num_games, self.initial_bankroll, dtype=cp.float32)
        
        # Game results storage
        results = []
        
        # Cut card position (4 decks from end)
        cut_position = len(self.full_shoe) - 104
        
        games_active = cp.ones(num_games, dtype=cp.bool_)
        
        while cp.any(games_active):
            # Check which games can continue
            can_play = (shoe_positions < cut_position - 10) & games_active
            
            if not cp.any(can_play):
                break
                
            # Calculate true counts
            cards_remaining = cut_position - shoe_positions
            decks_remaining = cards_remaining / 52.0
            true_counts = cp.where(decks_remaining > 0, running_counts / decks_remaining, 0)
            
            # Get betting amounts
            bet_amounts, num_hands = self.get_betting_amounts(true_counts)
            
            # Check bankroll constraints
            total_bet = bet_amounts * num_hands
            can_afford = bankrolls >= total_bet
            can_play = can_play & can_afford
            
            if not cp.any(can_play):
                break
            
            # Deal cards for active games
            active_indices = cp.where(can_play)[0]
            
            for game_idx in active_indices:
                game_idx = int(game_idx)
                
                if shoe_positions[game_idx] >= cut_position - 10:
                    # Reshuffle
                    cp.random.shuffle(shoes[game_idx])
                    shoe_positions[game_idx] = 0
                    running_counts[game_idx] = 0
                    continue
                
                # Play one hand
                result = self.play_single_hand_gpu(
                    shoes[game_idx], 
                    shoe_positions[game_idx], 
                    running_counts[game_idx],
                    bet_amounts[game_idx],
                    num_hands[game_idx]
                )
                
                if result is not None:
                    pos_change, count_change, game_result = result
                    shoe_positions[game_idx] += pos_change
                    running_counts[game_idx] += count_change
                    bankrolls[game_idx] += game_result
                    
                    # Store result
                    results.append({
                        'game_idx': game_idx,
                        'result': float(game_result),
                        'true_count': float(true_counts[game_idx]),
                        'bet_amount': float(bet_amounts[game_idx]),
                        'num_hands': int(num_hands[game_idx]),
                        'bankroll': float(bankrolls[game_idx])
                    })
                    
                    # Check if game should continue
                    if bankrolls[game_idx] <= 0:
                        games_active[game_idx] = False
        
        return results, bankrolls
    
    def play_single_hand_gpu(self, shoe, position, running_count, bet_amount, num_hands):
        """Play a single hand (simplified version for GPU)"""
        if position >= len(shoe) - 10:
            return None
            
        # Deal dealer upcard
        dealer_upcard = shoe[position]
        position += 1
        running_count += self.card_values[dealer_upcard]
        
        # Deal player cards
        player_cards = cp.zeros(4, dtype=cp.int32)  # Max 4 cards per hand
        player_cards[0] = shoe[position]
        player_cards[1] = shoe[position + 1]
        position += 2
        running_count += self.card_values[player_cards[0]]
        running_count += self.card_values[player_cards[1]]
        
        # Deal dealer hole card
        dealer_hole = shoe[position]
        position += 1
        running_count += self.card_values[dealer_hole]
        
        # Simplified play - just basic strategy hit/stand
        player_total = self.calculate_hand_value(player_cards.reshape(1, -1), 2)[0]
        
        # Hit until stand or bust
        card_count = 2
        while player_total < 17 and card_count < 4:
            if position >= len(shoe):
                return None
            player_cards[card_count] = shoe[position]
            position += 1
            running_count += self.card_values[player_cards[card_count]]
            card_count += 1
            player_total = self.calculate_hand_value(player_cards.reshape(1, -1), card_count)[0]
        
        # Play dealer
        dealer_cards = cp.array([dealer_upcard, dealer_hole])
        dealer_total = self.calculate_hand_value(dealer_cards.reshape(1, -1), 2)[0]
        dealer_card_count = 2
        
        while dealer_total < 17 and dealer_card_count < 6:
            if position >= len(shoe):
                return None
            new_card = shoe[position]
            position += 1
            running_count += self.card_values[new_card]
            dealer_total = int(self.bj_values[new_card]) + dealer_total
            if dealer_total > 21:
                break
            dealer_card_count += 1
        
        # Determine winner
        if player_total > 21:
            result = -bet_amount * num_hands
        elif dealer_total > 21:
            result = bet_amount * num_hands
        elif player_total > dealer_total:
            result = bet_amount * num_hands
        elif player_total < dealer_total:
            result = -bet_amount * num_hands
        else:
            result = 0
            
        return position - (4 + dealer_card_count - 2), running_count, result
    
    def run_simulation(self, num_games=10000, batch_size=None):
        """Run the full simulation"""
        if batch_size is None:
            batch_size = min(self.batch_size, num_games)
            
        all_results = []
        final_bankrolls = []
        
        print(f"Running GPU simulation with {num_games} games in batches of {batch_size}...")
        
        for batch_start in range(0, num_games, batch_size):
            batch_end = min(batch_start + batch_size, num_games)
            current_batch_size = batch_end - batch_start
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(num_games-1)//batch_size + 1}")
            
            batch_results, batch_bankrolls = self.play_batch_gpu(current_batch_size)
            all_results.extend(batch_results)
            final_bankrolls.extend(batch_bankrolls.tolist())
            
            # Clear GPU memory
            cp.get_default_memory_pool().free_all_blocks()
        
        return all_results, final_bankrolls
    
    def print_statistics(self, results, final_bankrolls):
        """Print comprehensive statistics"""
        if not results:
            print("No results to analyze")
            return
            
        results_array = np.array([(r['result'], r['true_count'], r['bet_amount'], r['num_hands']) 
                                 for r in results])
        
        total_games = len(set(r['game_idx'] for r in results))
        total_hands = len(results)
        total_wagered = np.sum(results_array[:, 2] * results_array[:, 3])
        total_won = np.sum(results_array[:, 0])
        
        avg_final_bankroll = np.mean(final_bankrolls)
        winning_games = np.sum(np.array(final_bankrolls) > self.initial_bankroll)
        
        print(f"\n=== GPU SIMULATION RESULTS ===")
        print(f"Total games: {total_games}")
        print(f"Total hands: {total_hands}")
        print(f"Total wagered: ${total_wagered:,.2f}")
        print(f"Total won/lost: ${total_won:,.2f}")
        print(f"House edge: {(-total_won/total_wagered)*100:.3f}%")
        print(f"Average final bankroll: ${avg_final_bankroll:,.2f}")
        print(f"Winning games: {winning_games}/{total_games} ({winning_games/total_games*100:.1f}%)")
        print(f"ROI: {((avg_final_bankroll-self.initial_bankroll)/self.initial_bankroll)*100:.2f}%")
        
        # True count analysis
        count_results = defaultdict(list)
        for result in results:
            tc = int(result['true_count'])
            count_results[tc].append(result['result'])
        
        print(f"\nResults by True Count:")
        for tc in sorted(count_results.keys()):
            if -5 <= tc <= 10:  # Reasonable range
                results_tc = count_results[tc]
                avg_result = np.mean(results_tc)
                print(f"TC {tc:2d}: {len(results_tc):5d} hands, avg: ${avg_result:6.2f}")

# GPU Memory and Performance Optimization
def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    # Set memory pool to use managed memory
    mempool = cp.get_default_memory_pool()
    mempool.set_limit(size=int(15.9 * 1024**3 * 0.8))  # Use 80% of available VRAM
    
    # Enable memory pool
    cp.cuda.MemoryPool().set_limit(size=int(15.9 * 1024**3 * 0.8))
    
    print(f"GPU Memory optimized for {cp.cuda.Device().mem_info[1] / 1024**3:.1f}GB VRAM")

if __name__ == "__main__":
    # Initialize GPU
    print("Initializing GPU...")
    print(f"GPU: {cp.cuda.get_device_name()}")
    print(f"CUDA Version: {cp.cuda.runtime.runtimeGetVersion()}")
    
    # Optimize memory
    optimize_gpu_memory()
    
    # Run simulation
    start_time = time.time()
    
    simulator = GPUBlackjackSimulator(batch_size=1000)
    results, final_bankrolls = simulator.run_simulation(num_games=50000)
    
    end_time = time.time()
    
    print(f"\nSimulation completed in {end_time - start_time:.2f} seconds")
    print(f"Performance: {len(results)/(end_time - start_time):.0f} hands/second")
    
    simulator.print_statistics(results, final_bankrolls)
    
    # GPU utilization info
    print(f"\nGPU Memory Usage: {cp.get_default_memory_pool().used_bytes()/1024**3:.1f}GB")
    print(f"GPU Memory Pool: {cp.get_default_memory_pool().total_bytes()/1024**3:.1f}GB")