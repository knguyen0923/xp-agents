"""THROWAWAY Showdown spike opponents (docs/superpowers/plans/2026-10-26-showdown-spike.md). Not part of xp/."""
from __future__ import annotations

import random
from poke_env.battle import DoubleBattle, Move
from poke_env.player import Player
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder, DefaultBattleOrder


class DamageHeuristicPlayer(Player):
    """Per active mon: best (move, foe target) by base power * accuracy * STAB * type
    effectiveness, weighted up when the target is low on HP; spread moves count both foes.
    Switches randomly only when no move is available. No protect, no switching strategy."""

    def choose_move(self, battle):
        if not isinstance(battle, DoubleBattle) or any(battle.force_switch):
            return self.choose_random_doubles_move(battle)
        foes = battle.opponent_active_pokemon
        orders, switched_in = [], None
        for mon, moves, switches in zip(battle.active_pokemon, battle.available_moves, battle.available_switches):
            switches = [s for s in switches if s != switched_in]
            if not mon or mon.fainted:
                orders.append(None); continue
            if not moves and switches:
                switched_in = random.choice(switches)
                orders.append(SingleBattleOrder(switched_in)); continue
            if not moves:
                orders.append(DefaultBattleOrder()); continue
            best, best_score = None, -1.0
            for move in moves:
                for tgt in battle.get_possible_showdown_targets(move, mon):
                    if tgt not in (1, 2):
                        continue
                    foe = foes[tgt - 1]
                    if foe is None or foe.fainted:
                        continue
                    score = self._score(move, mon, foe)
                    if score > best_score:
                        best, best_score = (move, tgt), score
            if best is None:
                move = random.choice(moves)
                tg = battle.get_possible_showdown_targets(move, mon)
                orders.append(SingleBattleOrder(move, move_target=random.choice(tg)))
            else:
                orders.append(SingleBattleOrder(best[0], move_target=best[1]))
        for slot, order in enumerate(orders):
            # one Mega per side per battle: take the first slot that can
            if isinstance(order, SingleBattleOrder) and isinstance(order.order, Move) and battle.can_mega_evolve[slot]:
                order.mega = True
                break
        if orders[0] or orders[1]:
            return DoubleBattleOrder(orders[0], orders[1])
        return self.choose_random_doubles_move(battle)

    @staticmethod
    def _score(move, user, foe):
        if move.base_power <= 0:
            return 0.0
        s = move.base_power * (move.accuracy if move.accuracy else 1.0)
        if move.type in (user.type_1, user.type_2):
            s *= 1.5
        s *= foe.damage_multiplier(move)
        s *= 2.0 - foe.current_hp_fraction
        return s
