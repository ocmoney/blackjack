import random
import math
from collections import defaultdict

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
    
    def value(self):
        if self.rank in ['J', 'Q', 'K']:
            return 10
        elif self.rank == 'A':
            return 11
        else:
            return int(self.rank)
    
    def hi_lo_value(self):
        if self.rank in ['2', '3', '4', '5', '6']:
            return 1
        elif self.rank in ['10', 'J', 'Q', 'K', 'A']:
            return -1
        else:
            return 0
    
    def __str__(self):
        return f"{self.rank}{self.suit}"

class Hand:
    def __init__(self):
        self.cards = []
        self.bet = 0
        self.doubled = False
        self.split_from = None
        self.surrendered = False
    
    def add_card(self, card):
        self.cards.append(card)
    
    def value(self):
        total = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')
        
        while total > 21 and aces > 0:
            total -= 10
            aces -= 1
        
        return total
    
    def is_blackjack(self):
        return len(self.cards) == 2 and self.value() == 21
    
    def is_bust(self):
        return self.value() > 21
    
    def is_soft(self):
        total = sum(card.value() for card in self.cards)
        aces = sum(1 for card in self.cards if card.rank == 'A')
        return total > 21 and aces > 0
    
    def can_split(self):
        return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank
    
    def can_double(self):
        return len(self.cards) == 2 and not self.split_from

class Shoe:
    def __init__(self, num_decks=6):
        self.num_decks = num_decks
        self.cards = []
        self.cut_card_position = 0
        self.reset()
    
    def reset(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        
        self.cards = []
        for _ in range(self.num_decks):
            for rank in ranks:
                for suit in suits:
                    self.cards.append(Card(rank, suit))
        
        random.shuffle(self.cards)
        # Cut off 2 decks (104 cards) from the back
        self.cut_card_position = len(self.cards) - 104
    
    def deal_card(self):
        if len(self.cards) <= self.cut_card_position:
            return None
        return self.cards.pop()
    
    def cards_remaining(self):
        return len(self.cards) - self.cut_card_position
    
    def penetration(self):
        total_cards = self.num_decks * 52
        return (total_cards - len(self.cards)) / total_cards

class BasicStrategy:
    def __init__(self):
        # Hard totals against dealer upcard
        self.hard_strategy = {
            # Player total: {dealer_upcard: action}
            5: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            6: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            7: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            8: {2:'H', 3:'H', 4:'H', 5:'H', 6:'H', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            9: {2:'H', 3:'D', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            10: {2:'D', 3:'D', 4:'D', 5:'D', 6:'D', 7:'D', 8:'D', 9:'D', 10:'H', 'A':'H'},
            11: {2:'D', 3:'D', 4:'D', 5:'D', 6:'D', 7:'D', 8:'D', 9:'D', 10:'D', 'A':'H'},
            12: {2:'H', 3:'H', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            13: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            14: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            15: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            16: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            17: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            18: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            19: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            20: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            21: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
        }
        
        # Soft totals against dealer upcard
        self.soft_strategy = {
            # Soft total: {dealer_upcard: action}
            13: {2:'H', 3:'H', 4:'H', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            14: {2:'H', 3:'H', 4:'H', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            15: {2:'H', 3:'H', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            16: {2:'H', 3:'H', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            17: {2:'H', 3:'D', 4:'D', 5:'D', 6:'D', 7:'H', 8:'H', 9:'H', 10:'H', 'A':'H'},
            18: {2:'S', 3:'D', 4:'D', 5:'D', 6:'D', 7:'S', 8:'S', 9:'H', 10:'H', 'A':'H'},
            19: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            20: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
            21: {2:'S', 3:'S', 4:'S', 5:'S', 6:'S', 7:'S', 8:'S', 9:'S', 10:'S', 'A':'S'},
        }
        
        # Pair splitting strategy
        self.pair_strategy = {
            # Pair: {dealer_upcard: action}
            'A': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'Y', 8:'Y', 9:'Y', 10:'Y', 'A':'Y'},
            '2': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'Y', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '3': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'Y', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '4': {2:'N', 3:'N', 4:'N', 5:'Y', 6:'Y', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '5': {2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '6': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '7': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'Y', 8:'N', 9:'N', 10:'N', 'A':'N'},
            '8': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'Y', 8:'Y', 9:'Y', 10:'Y', 'A':'Y'},
            '9': {2:'Y', 3:'Y', 4:'Y', 5:'Y', 6:'Y', 7:'N', 8:'Y', 9:'Y', 10:'N', 'A':'N'},
            '10': {2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            'J': {2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            'Q': {2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
            'K': {2:'N', 3:'N', 4:'N', 5:'N', 6:'N', 7:'N', 8:'N', 9:'N', 10:'N', 'A':'N'},
        }
    
    def get_action(self, hand, dealer_upcard, true_count=0):
        dealer_card = dealer_upcard.rank if dealer_upcard.rank != '10' else 10
        if dealer_card in ['J', 'Q', 'K']:
            dealer_card = 10
        
        # Count variations for specific situations
        if self.should_deviate_from_basic_strategy(hand, dealer_upcard, true_count):
            return self.get_index_play(hand, dealer_upcard, true_count)
        
        # Check for pair splitting first
        if hand.can_split():
            pair_rank = hand.cards[0].rank
            if pair_rank in self.pair_strategy:
                action = self.pair_strategy[pair_rank].get(dealer_card, 'N')
                if action == 'Y':
                    return 'P'
        
        # Check if hand is soft
        if hand.is_soft():
            hand_value = hand.value()
            if hand_value in self.soft_strategy:
                action = self.soft_strategy[hand_value].get(dealer_card, 'H')
                if action == 'D' and not hand.can_double():
                    return 'H'
                return action
        
        # Hard totals
        hand_value = hand.value()
        if hand_value in self.hard_strategy:
            action = self.hard_strategy[hand_value].get(dealer_card, 'H')
            if action == 'D' and not hand.can_double():
                return 'H'
            return action
        
        return 'S' if hand_value >= 17 else 'H'
    
    def should_deviate_from_basic_strategy(self, hand, dealer_upcard, true_count):
        # Common index plays (simplified)
        if true_count == 0:
            return False
        
        hand_value = hand.value()
        dealer_card = dealer_upcard.rank if dealer_upcard.rank != '10' else 10
        if dealer_card in ['J', 'Q', 'K']:
            dealer_card = 10
        
        # Some common deviations
        if hand_value == 16 and dealer_card == 10 and true_count >= 0:
            return True
        if hand_value == 15 and dealer_card == 10 and true_count >= 4:
            return True
        if hand_value == 12 and dealer_card == 3 and true_count >= 2:
            return True
        if hand_value == 12 and dealer_card == 2 and true_count >= 3:
            return True
        
        return False
    
    def get_index_play(self, hand, dealer_upcard, true_count):
        hand_value = hand.value()
        dealer_card = dealer_upcard.rank if dealer_upcard.rank != '10' else 10
        if dealer_card in ['J', 'Q', 'K']:
            dealer_card = 10
        
        # Index play deviations
        if hand_value == 16 and dealer_card == 10 and true_count >= 0:
            return 'S'
        if hand_value == 15 and dealer_card == 10 and true_count >= 4:
            return 'S'
        if hand_value == 12 and dealer_card == 3 and true_count >= 2:
            return 'S'
        if hand_value == 12 and dealer_card == 2 and true_count >= 3:
            return 'S'
        
        # Default to basic strategy if no deviation
        return self.get_action(hand, dealer_upcard, 0)

class BlackjackSimulator:
    def __init__(self):
        self.shoe = Shoe(6)
        self.strategy = BasicStrategy()
        self.running_count = 0
        self.cards_seen = 0
        self.bankroll = 10000
        self.min_bet = 25
        self.results = []
        
    def true_count(self):
        decks_remaining = self.shoe.cards_remaining() / 52
        if decks_remaining <= 0:
            return 0
        return self.running_count / decks_remaining
    
    def get_bet_amount(self, true_count):
        # Betting spread: 25, 100, 200, 400, 2x400, 2x500
        if true_count < 1:
            return 25, 1  # bet amount, number of hands
        elif true_count < 2:
            return 100, 1
        elif true_count < 3:
            return 200, 1
        elif true_count < 4:
            return 400, 1
        elif true_count < 5:
            return 400, 2
        else:
            return 500, 2
    
    def update_count(self, card):
        self.running_count += card.hi_lo_value()
        self.cards_seen += 1
    
    def deal_hand(self):
        hand = Hand()
        for _ in range(2):
            card = self.shoe.deal_card()
            if card is None:
                return None
            hand.add_card(card)
            self.update_count(card)
        return hand
    
    def play_dealer_hand(self, dealer_hand):
        while dealer_hand.value() < 17:
            card = self.shoe.deal_card()
            if card is None:
                return False
            dealer_hand.add_card(card)
            self.update_count(card)
        return True
    
    def play_hand(self, hand, dealer_upcard):
        while True:
            if hand.is_bust():
                return 'bust'
            
            if hand.value() == 21:
                return 'stand'
            
            action = self.strategy.get_action(hand, dealer_upcard, self.true_count())
            
            if action == 'H':  # Hit
                card = self.shoe.deal_card()
                if card is None:
                    return 'shoe_empty'
                hand.add_card(card)
                self.update_count(card)
            
            elif action == 'S':  # Stand
                return 'stand'
            
            elif action == 'D':  # Double
                if hand.can_double():
                    card = self.shoe.deal_card()
                    if card is None:
                        return 'shoe_empty'
                    hand.add_card(card)
                    self.update_count(card)
                    hand.doubled = True
                    return 'double'
                else:
                    # Can't double, so hit
                    card = self.shoe.deal_card()
                    if card is None:
                        return 'shoe_empty'
                    hand.add_card(card)
                    self.update_count(card)
            
            elif action == 'P':  # Split
                return 'split'
    
    def calculate_result(self, player_hands, dealer_hand):
        dealer_value = dealer_hand.value()
        dealer_blackjack = dealer_hand.is_blackjack()
        dealer_bust = dealer_hand.is_bust()
        
        total_result = 0
        
        for hand in player_hands:
            bet = hand.bet
            
            if hand.is_bust():
                result = -bet
            elif hand.surrendered:
                result = -bet * 0.5
            elif hand.is_blackjack() and not dealer_blackjack:
                result = bet * 1.5
            elif dealer_bust and not hand.is_bust():
                result = bet * (2 if hand.doubled else 1)
            elif hand.value() > dealer_value:
                result = bet * (2 if hand.doubled else 1)
            elif hand.value() < dealer_value:
                result = -bet
            else:  # Tie
                result = 0
            
            total_result += result
        
        return total_result
    
    def play_round(self):
        if self.shoe.cards_remaining() < 50:
            return None
        
        tc = self.true_count()
        bet_amount, num_hands = self.get_bet_amount(tc)
        
        # Check if we have enough bankroll
        if self.bankroll < bet_amount * num_hands:
            return None
        
        # Deal dealer upcard
        dealer_hand = Hand()
        dealer_upcard = self.shoe.deal_card()
        if dealer_upcard is None:
            return None
        dealer_hand.add_card(dealer_upcard)
        self.update_count(dealer_upcard)
        
        # Deal player hands
        player_hands = []
        for _ in range(num_hands):
            hand = self.deal_hand()
            if hand is None:
                return None
            hand.bet = bet_amount
            player_hands.append(hand)
        
        # Deal dealer hole card
        dealer_hole_card = self.shoe.deal_card()
        if dealer_hole_card is None:
            return None
        dealer_hand.add_card(dealer_hole_card)
        self.update_count(dealer_hole_card)
        
        # Play each hand
        final_hands = []
        for hand in player_hands:
            hands_to_play = [hand]
            
            while hands_to_play:
                current_hand = hands_to_play.pop(0)
                result = self.play_hand(current_hand, dealer_upcard)
                
                if result == 'shoe_empty':
                    return None
                elif result == 'split':
                    # Split the hand
                    if len(final_hands) + len(hands_to_play) < 4:  # Limit splits
                        new_hand = Hand()
                        new_hand.add_card(current_hand.cards.pop())
                        new_hand.bet = current_hand.bet
                        
                        # Deal new cards
                        card1 = self.shoe.deal_card()
                        card2 = self.shoe.deal_card()
                        if card1 is None or card2 is None:
                            return None
                        
                        current_hand.add_card(card1)
                        new_hand.add_card(card2)
                        self.update_count(card1)
                        self.update_count(card2)
                        
                        hands_to_play.append(current_hand)
                        hands_to_play.append(new_hand)
                    else:
                        # Can't split more, play as normal
                        hands_to_play.append(current_hand)
                else:
                    final_hands.append(current_hand)
        
        # Play dealer hand
        if not self.play_dealer_hand(dealer_hand):
            return None
        
        # Calculate results
        result = self.calculate_result(final_hands, dealer_hand)
        self.bankroll += result
        
        return {
            'result': result,
            'true_count': tc,
            'bet_amount': bet_amount,
            'num_hands': num_hands,
            'bankroll': self.bankroll
        }
    
    def run_simulation(self, num_rounds=1000):
        round_count = 0
        
        while round_count < num_rounds and self.bankroll > 0:
            # Check if we need a new shoe
            if self.shoe.cards_remaining() < 50:
                self.shoe.reset()
                self.running_count = 0
                self.cards_seen = 0
            
            result = self.play_round()
            if result is None:
                continue
            
            self.results.append(result)
            round_count += 1
            
            if round_count % 100 == 0:
                print(f"Round {round_count}: Bankroll = ${self.bankroll:.2f}, True Count = {result['true_count']:.1f}")
        
        return self.results
    
    def print_statistics(self):
        if not self.results:
            print("No results to analyze")
            return
        
        total_rounds = len(self.results)
        total_wagered = sum(r['bet_amount'] * r['num_hands'] for r in self.results)
        total_won = sum(r['result'] for r in self.results)
        
        print(f"\n=== SIMULATION RESULTS ===")
        print(f"Total rounds played: {total_rounds}")
        print(f"Total wagered: ${total_wagered:.2f}")
        print(f"Total won/lost: ${total_won:.2f}")
        print(f"House edge: {(-total_won/total_wagered)*100:.3f}%")
        print(f"Final bankroll: ${self.bankroll:.2f}")
        print(f"Return on investment: {((self.bankroll-10000)/10000)*100:.2f}%")
        
        # Count distribution
        count_results = defaultdict(list)
        for result in self.results:
            tc = int(result['true_count'])
            count_results[tc].append(result['result'])
        
        print(f"\nResults by True Count:")
        for tc in sorted(count_results.keys()):
            results = count_results[tc]
            avg_result = sum(results) / len(results)
            print(f"TC {tc}: {len(results)} hands, avg result: ${avg_result:.2f}")

# Run simulation
if __name__ == "__main__":
    sim = BlackjackSimulator()
    print("Starting blackjack simulation...")
    print(f"Initial bankroll: ${sim.bankroll}")
    print(f"Betting spread: TC<1: ${sim.min_bet}, TC1: $100, TC2: $200, TC3: $400, TC4: 2x$400, TC5+: 2x$500")
    
    results = sim.run_simulation(1000)
    sim.print_statistics()