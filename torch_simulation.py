import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
import time
from tqdm import tqdm

class GPUBlackjackSimulator:
    def __init__(self, batch_size=1000, device='cuda'):
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_decks = 6
        self.min_bet = 25
        self.initial_bankroll = 100000
        
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # Initialize GPU tensors
        self.setup_gpu_tensors()
        self.setup_strategy_tables()
        
    def setup_gpu_tensors(self):
        """Initialize GPU tensors for batch processing"""
        # Card values for Hi-Lo counting: 2-6: +1, 7-9: 0, 10-A: -1
        self.card_values = torch.tensor([
            1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1  # indices 0-12 for cards 2-A
        ], dtype=torch.int32, device=self.device)
        
        # Blackjack values: 2-A
        self.bj_values = torch.tensor([
            2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11  # indices 0-12 for cards 2-A
        ], dtype=torch.int32, device=self.device)
        
        # Create a single deck (52 cards)
        single_deck = torch.arange(13, dtype=torch.int32, device=self.device).repeat(4)
        self.full_shoe = single_deck.repeat(self.num_decks)
        
        print(f"Initialized {len(self.full_shoe)} card shoe on {self.device}")
        
    def setup_strategy_tables(self):
        """Setup basic strategy tables on GPU"""
        # Basic strategy: 0=Hit, 1=Stand, 2=Double, 3=Split
        
        # Hard totals (player 5-21 vs dealer 2-A)
        hard_data = [
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
        ]
        
        self.hard_strategy = torch.tensor(hard_data, dtype=torch.int32, device=self.device)
        
        # Soft totals (A,2 to A,9 vs dealer 2-A)
        soft_data = [
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],  # A,2 (soft 13)
            [0, 0, 0, 2, 2, 0, 0, 0, 0, 0],  # A,3 (soft 14)
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],  # A,4 (soft 15)
            [0, 0, 2, 2, 2, 0, 0, 0, 0, 0],  # A,5 (soft 16)
            [0, 2, 2, 2, 2, 0, 0, 0, 0, 0],  # A,6 (soft 17)
            [1, 2, 2, 2, 2, 1, 1, 0, 0, 0],  # A,7 (soft 18)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,8 (soft 19)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,9 (soft 20)
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # A,10 (soft 21)
        ]
        
        self.soft_strategy = torch.tensor(soft_data, dtype=torch.int32, device=self.device)
        
        # Pair splitting (2,2 to A,A vs dealer 2-A)
        pair_data = [
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
        ]
        
        self.pair_strategy = torch.tensor(pair_data, dtype=torch.int32, device=self.device)
        
    def shuffle_shoes(self, num_games):
        """Generate shuffled shoes for batch processing"""
        batch_shoes = []
        shoe_length = len(self.full_shoe)
        
        for _ in range(num_games):
            # Create a random permutation
            indices = torch.randperm(shoe_length, device=self.device)
            shuffled_shoe = self.full_shoe[indices]
            batch_shoes.append(shuffled_shoe)
        
        return torch.stack(batch_shoes)
    
    def calculate_hand_value(self, cards):
        """Calculate hand values for batch of hands"""
        # Convert card indices to blackjack values
        values = self.bj_values[cards]
        
        # Handle -1 (no card) by setting to 0
        values = torch.where(cards == -1, 0, values)
        
        # Sum values
        total = torch.sum(values, dim=-1)
        
        # Count aces
        aces = torch.sum(cards == 12, dim=-1)  # Ace is index 12
        
        # Adjust for aces (convert from 11 to 1 if needed)
        while torch.any((total > 21) & (aces > 0)):
            mask = (total > 21) & (aces > 0)
            total = torch.where(mask, total - 10, total)
            aces = torch.where(mask, aces - 1, aces)
        
        return total
    
    def get_betting_amounts(self, true_counts):
        """Calculate betting amounts and number of hands based on true count"""
        # Betting progression based on true count:
        # 25 for tc<1, 100 for 2>tc>=1, 200 for 3>tc>=2, 400 for 4>tc>=3, 2 x 400 for 5>tc>=4, 2 x 500 for tc>=5
        bet_amounts = torch.where(true_counts < 1, 25,
                     torch.where(true_counts < 2, 100,
                     torch.where(true_counts < 3, 200,
                     torch.where(true_counts < 4, 400,
                     torch.where(true_counts < 5, 400, 500)))))
        
        # Number of hands (1 for TC<4, 2 for TC>=4)
        num_hands = torch.where(true_counts < 4, 1, 2)
        
        return bet_amounts, num_hands
    
    def get_basic_strategy_action(self, player_cards, dealer_upcard, true_count):
        """Get basic strategy action for batch of hands"""
        batch_size = player_cards.shape[0]
        
        # Calculate hand values
        hand_values = self.calculate_hand_value(player_cards)
        
        # Check for pairs
        is_pair = (player_cards[:, 0] == player_cards[:, 1]) & (player_cards[:, 0] != -1)
        
        # Check for soft hands (has ace counted as 11)
        has_ace = torch.any(player_cards == 12, dim=1)
        card_sum = torch.sum(torch.where(player_cards == -1, 0, 
                           torch.where(player_cards == 12, 1, self.bj_values[player_cards])), dim=1)
        is_soft = has_ace & (card_sum + 10 == hand_values)
        
        # Convert dealer upcard to strategy table index (0-9 for 2-A)
        dealer_idx = torch.where(dealer_upcard == 12, 9,  # Ace -> 9
                    torch.where(dealer_upcard >= 9, 8,     # 10,J,Q,K -> 8
                    dealer_upcard))                        # 2-9 -> 0-7
        
        # Get actions
        actions = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
        
        # Pair splitting
        pair_mask = is_pair & (player_cards[:, 0] < 13)
        if torch.any(pair_mask):
            pair_indices = player_cards[pair_mask, 0]
            pair_actions = self.pair_strategy[pair_indices, dealer_idx[pair_mask]]
            actions[pair_mask] = pair_actions
        
        # Soft hands
        soft_mask = is_soft & ~pair_mask & (hand_values >= 13) & (hand_values <= 21)
        if torch.any(soft_mask):
            soft_indices = hand_values[soft_mask] - 13
            soft_actions = self.soft_strategy[soft_indices, dealer_idx[soft_mask]]
            actions[soft_mask] = soft_actions
        
        # Hard hands
        hard_mask = ~is_soft & ~pair_mask & (hand_values >= 5) & (hand_values <= 21)
        if torch.any(hard_mask):
            hard_indices = hand_values[hard_mask] - 5
            hard_actions = self.hard_strategy[hard_indices, dealer_idx[hard_mask]]
            actions[hard_mask] = hard_actions
        
        # Count variations (simplified)
        # 16 vs 10: Stand if TC >= 0
        count_mask = (hand_values == 16) & (dealer_upcard >= 9) & (true_count >= 0)
        actions[count_mask] = 1  # Stand
        
        # 15 vs 10: Stand if TC >= 4
        count_mask = (hand_values == 15) & (dealer_upcard >= 9) & (true_count >= 4)
        actions[count_mask] = 1  # Stand
        
        return actions
    
    def play_batch_parallel(self, num_games):
        """Play multiple games in parallel on GPU using vectorized operations"""
        print(f"Playing {num_games} games in parallel on GPU...")
        
        # Initialize game states
        shoes = self.shuffle_shoes(num_games)
        shoe_positions = torch.zeros(num_games, dtype=torch.int32, device=self.device)
        running_counts = torch.zeros(num_games, dtype=torch.int32, device=self.device)
        bankrolls = torch.full((num_games,), self.initial_bankroll, dtype=torch.float32, device=self.device)
        
        # Game results
        all_results = []
        
        # Cut card position (1 deck from end for more hands)
        cut_position = len(self.full_shoe) - 52
        max_hands_per_game = 40
        
        # Progress bar for hands
        pbar = tqdm(range(max_hands_per_game), desc=f"Playing {num_games} games", unit="hands")
        
        for hand_num in pbar:
            # Check which games can continue
            can_play = (shoe_positions < cut_position - 10) & (bankrolls > 0)
            
            if not torch.any(can_play):
                break
            
            # Reshuffle shoes that are too low (vectorized)
            need_shuffle = shoe_positions >= cut_position - 10
            if torch.any(need_shuffle):
                shuffle_indices = torch.where(need_shuffle)[0]
                for i in shuffle_indices:
                    indices = torch.randperm(len(self.full_shoe), device=self.device)
                    shoes[i] = self.full_shoe[indices]
                    shoe_positions[i] = 0
                    running_counts[i] = 0
            
            # Calculate true counts (vectorized)
            cards_remaining = cut_position - shoe_positions
            decks_remaining = cards_remaining.float() / 52.0
            true_counts = torch.where(decks_remaining > 0, 
                                    running_counts.float() / decks_remaining, 
                                    torch.zeros_like(running_counts.float()))
            
            # Get betting amounts (vectorized)
            bet_amounts, num_hands = self.get_betting_amounts(true_counts)
            total_bets = bet_amounts.float() * num_hands.float()
            
            # Filter games that can afford the bet
            can_play = can_play & (bankrolls >= total_bets)
            
            if not torch.any(can_play):
                continue
            
            # VECTORIZED GAME PROCESSING - Process all active games simultaneously
            
            # Get active game indices
            active_mask = can_play
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) == 0:
                continue
            
            # Deal cards for all active games simultaneously
            # Create batch tensors for all active games
            batch_size = len(active_indices)
            
            # Get current positions for all active games
            current_positions = shoe_positions[active_indices]
            
            # Check if we have enough cards
            valid_positions = current_positions < len(self.full_shoe) - 10
            if not torch.any(valid_positions):
                continue
            
            # Filter to valid games
            valid_indices = active_indices[valid_positions]
            valid_positions_tensor = current_positions[valid_positions]
            
            if len(valid_indices) == 0:
                continue
            
            # Deal cards for all valid games at once
            # Dealer upcards
            dealer_upcards = shoes[valid_indices, valid_positions_tensor]
            
            # Player cards (first two)
            player_card1 = shoes[valid_indices, valid_positions_tensor + 1]
            player_card2 = shoes[valid_indices, valid_positions_tensor + 2]
            
            # Dealer hole cards
            dealer_holes = shoes[valid_indices, valid_positions_tensor + 3]
            
            # Update running counts for initial cards
            count_changes = (self.card_values[dealer_upcards] + 
                           self.card_values[player_card1] + 
                           self.card_values[player_card2] + 
                           self.card_values[dealer_holes])
            
            # Calculate initial hand values
            player_cards_batch = torch.stack([player_card1, player_card2], dim=1)
            player_values = self.calculate_hand_value(player_cards_batch)
            
            dealer_cards_batch = torch.stack([dealer_upcards, dealer_holes], dim=1)
            dealer_values = self.calculate_hand_value(dealer_cards_batch)
            
            # Track card positions
            card_positions = valid_positions_tensor + 4
            
            # Play player hands (simplified strategy - hit until 17+)
            max_player_cards = 4
            for card_idx in range(2, max_player_cards):
                # Check which games need more cards
                need_more_cards = (player_values < 17) & (card_positions < len(self.full_shoe))
                
                if not torch.any(need_more_cards):
                    break
                
                # Deal additional cards
                new_cards = shoes[valid_indices, card_positions]
                count_changes += torch.where(need_more_cards, self.card_values[new_cards], 0)
                
                # Update player hands
                player_cards_batch = torch.cat([
                    player_cards_batch, 
                    new_cards.unsqueeze(1)
                ], dim=1)
                
                # Recalculate player values
                player_values = self.calculate_hand_value(player_cards_batch)
                card_positions += 1
            
            # Play dealer hands (hit until 17+)
            max_dealer_cards = 6
            for card_idx in range(2, max_dealer_cards):
                # Check which games need dealer to hit
                dealer_needs_hit = (dealer_values < 17) & (card_positions < len(self.full_shoe))
                
                if not torch.any(dealer_needs_hit):
                    break
                
                # Deal additional cards to dealer
                new_cards = shoes[valid_indices, card_positions]
                count_changes += torch.where(dealer_needs_hit, self.card_values[new_cards], 0)
                
                # Update dealer hands
                dealer_cards_batch = torch.cat([
                    dealer_cards_batch, 
                    new_cards.unsqueeze(1)
                ], dim=1)
                
                # Recalculate dealer values
                dealer_values = self.calculate_hand_value(dealer_cards_batch)
                card_positions += 1
            
            # Determine results (vectorized)
            bet_amounts_valid = bet_amounts[valid_indices]
            num_hands_valid = num_hands[valid_indices]
            
            # Calculate results
            player_bust = player_values > 21
            dealer_bust = dealer_values > 21
            player_wins = (player_values > dealer_values) & ~player_bust
            dealer_wins = (dealer_values > player_values) & ~dealer_bust
            push = (player_values == dealer_values) & ~player_bust & ~dealer_bust
            
            # Calculate payouts
            payouts = torch.zeros_like(bet_amounts_valid, dtype=torch.float32)
            # Convert num_hands to float32 to match bet_amounts dtype
            num_hands_float = num_hands_valid.float()
            payouts[dealer_bust] = bet_amounts_valid[dealer_bust] * num_hands_float[dealer_bust]
            payouts[player_wins] = bet_amounts_valid[player_wins] * num_hands_float[player_wins]
            payouts[player_bust] = -bet_amounts_valid[player_bust] * num_hands_float[player_bust]
            payouts[dealer_wins] = -bet_amounts_valid[dealer_wins] * num_hands_float[dealer_wins]
            # push = 0 (already initialized)
            
            # Update game states
            shoe_positions[valid_indices] = card_positions
            running_counts[valid_indices] += count_changes
            bankrolls[valid_indices] += payouts
            
            # Store results (batch operation)
            true_counts_valid = true_counts[valid_indices]
            bankrolls_valid = bankrolls[valid_indices]
            
            # Convert to Python for storage (this is the only sequential part, but much smaller)
            for i in range(len(valid_indices)):
                all_results.append({
                    'game_idx': int(valid_indices[i]),
                    'result': float(payouts[i]),
                    'true_count': float(true_counts_valid[i]),
                    'bet_amount': float(bet_amounts_valid[i]),
                    'num_hands': int(num_hands_valid[i]),
                    'bankroll': float(bankrolls_valid[i])
                })
            
            # Update progress bar with real-time stats
            active_count = torch.sum(can_play).item()
            avg_bankroll = torch.mean(bankrolls).item()
            pbar.set_postfix({
                'Active': active_count,
                'Avg Bankroll': f'${avg_bankroll:.0f}',
                'Hands': len(all_results)
            })
            
            # Stop if no games can continue
            if not torch.any(can_play):
                break
        
        return all_results, bankrolls.cpu().numpy()
    
    def run_simulation(self, num_games=10000):
        """Run the complete simulation"""
        print(f"\nStarting GPU Blackjack Simulation")
        print(f"Games: {num_games}")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        
        start_time = time.time()
        
        # Run in batches to manage memory
        all_results = []
        all_bankrolls = []
        
        num_batches = (num_games - 1) // self.batch_size + 1
        batch_pbar = tqdm(range(0, num_games, self.batch_size), 
                         desc="Processing batches", 
                         unit="batch")
        
        for batch_start in batch_pbar:
            batch_end = min(batch_start + self.batch_size, num_games)
            current_batch_size = batch_end - batch_start
            
            batch_pbar.set_postfix({
                'Batch': f"{batch_start//self.batch_size + 1}/{num_batches}",
                'Size': current_batch_size,
                'Total Results': len(all_results)
            })
            
            batch_results, batch_bankrolls = self.play_batch_parallel(current_batch_size)
            all_results.extend(batch_results)
            all_bankrolls.extend(batch_bankrolls)
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"\nSimulation completed in {elapsed:.2f} seconds")
        print(f"Performance: {len(all_results)/elapsed:.0f} hands/second")
        
        return all_results, all_bankrolls
    
    def print_statistics(self, results, final_bankrolls):
        """Print detailed statistics"""
        if not results:
            print("No results to analyze")
            return
        
        total_games = len(final_bankrolls)
        total_hands = len(results)
        total_wagered = sum(r['bet_amount'] * r['num_hands'] for r in results)
        total_won = sum(r['result'] for r in results)
        
        avg_bankroll = np.mean(final_bankrolls)
        winning_games = sum(1 for b in final_bankrolls if b > self.initial_bankroll)
        
        print(f"\n{'='*50}")
        print(f"GPU BLACKJACK SIMULATION RESULTS")
        print(f"{'='*50}")
        print(f"Total games: {total_games:,}")
        print(f"Total hands: {total_hands:,}")
        print(f"Total wagered: ${total_wagered:,.2f}")
        print(f"Total won/lost: ${total_won:,.2f}")
        print(f"House edge: {(-total_won/total_wagered)*100:.3f}%")
        print(f"Average final bankroll: ${avg_bankroll:,.2f}")
        print(f"Winning games: {winning_games:,}/{total_games:,} ({winning_games/total_games*100:.1f}%)")
        print(f"ROI: {((avg_bankroll-self.initial_bankroll)/self.initial_bankroll)*100:.2f}%")
        
        # Bankroll distribution
        bankroll_ranges = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, float('inf'))]
        print(f"\nBankroll Distribution:")
        for low, high in bankroll_ranges:
            count = sum(1 for b in final_bankrolls if low <= b < high)
            if high == float('inf'):
                print(f"${low:,}+: {count:,} ({count/total_games*100:.1f}%)")
            else:
                print(f"${low:,}-${high:,}: {count:,} ({count/total_games*100:.1f}%)")
        
        # True count analysis
        count_results = defaultdict(list)
        for r in results:
            tc = int(r['true_count'])
            count_results[tc].append(r['result'])
        
        print(f"\nResults by True Count:")
        for tc in sorted(count_results.keys()):
            if -5 <= tc <= 10:
                tc_results = count_results[tc]
                avg_result = np.mean(tc_results)
                print(f"TC {tc:2d}: {len(tc_results):6,} hands, avg: ${avg_result:7.2f}")

if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("CUDA not available, using CPU")
    
    # Run simulation
    simulator = GPUBlackjackSimulator(batch_size=1000)
    results, bankrolls = simulator.run_simulation(num_games=10000)
    simulator.print_statistics(results, bankrolls)