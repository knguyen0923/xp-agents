"""Scripted opponents for the Showdown phase: the 3-rung ladder Random < MaxBasePower < DamageHeuristic, plus the tempo-aware TempoPlayer.

Used through `SingleAgentWrapper`, which only calls `choose_move`, so the players never open a connection.
"""
from __future__ import annotations

import random
from collections import Counter

from poke_env.battle import DoubleBattle, Field, Move, MoveCategory, SideCondition, Weather
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer
from poke_env.player.battle_order import DefaultBattleOrder, DoubleBattleOrder, SingleBattleOrder

LADDER = ("random", "maxbp", "damage")
OPPONENTS = LADDER + ("tempo",)


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


class TempoPlayer(DamageHeuristicPlayer):
    """The damage heuristic plus field management (spec 2026-11-02): before attacking, a slot may set Trick Room, Tailwind
    or a screen by fixed rules. At most one setter per slot per turn, and a condition is never set twice on one turn.
    Counters (`setter_counts`, `battles_seen`, `battles_with_setter`) let a run report how often it actually sets anything."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setter_counts: Counter = Counter()
        self.battles_seen: set = set()
        self.battles_with_setter: set = set()

    def choose_move(self, battle):
        order = super().choose_move(battle)
        self.battles_seen.add(battle.battle_tag)
        if any(battle.force_switch) or not isinstance(order, DoubleBattleOrder):
            return order
        slots, used = [order.first_order, order.second_order], set()
        for pos in range(2):
            move = self._setter(battle, pos, used)
            if move is None:
                continue
            old = slots[pos]
            slots[pos] = SingleBattleOrder(move, mega=bool(getattr(old, "mega", False)))   # a setter targets nobody (move_target 0)
            self.setter_counts[move.id] += 1
            self.battles_with_setter.add(battle.battle_tag)
        return DoubleBattleOrder(slots[0], slots[1])

    def _setter(self, battle, pos, used):
        """The setter move this slot should use now, or None. Adds the chosen condition to `used`."""
        from xp.showdown.tempo import effective_speed   # lazy: tempo.py imports encode.py

        mon = battle.active_pokemon[pos]
        if mon is None or mon.fainted:
            return None
        avail = {m.id: m for m in battle.available_moves[pos]}
        mine = [m for m in battle.active_pokemon if m is not None and not m.fainted]
        foes = [f for f in battle.opponent_active_pokemon if f is not None and not f.fainted]
        trick_room = Field.TRICK_ROOM in battle.fields

        def pick(move_id):
            used.add(move_id)
            return avail[move_id]

        if "trickroom" in avail and "trickroom" not in used and not trick_room and foes:
            mine_speed = sum(effective_speed(m, SideCondition.TAILWIND in battle.side_conditions, False) for m in mine) / len(mine)
            foe_speed = sum(effective_speed(f, SideCondition.TAILWIND in battle.opponent_side_conditions, True) for f in foes) / len(foes)
            if mine_speed < foe_speed:
                return pick("trickroom")
        if "tailwind" in avail and "tailwind" not in used and SideCondition.TAILWIND not in battle.side_conditions and not trick_room:
            return pick("tailwind")
        if ("auroraveil" in avail and "auroraveil" not in used and SideCondition.AURORA_VEIL not in battle.side_conditions
                and (Weather.SNOWSCAPE in battle.weather or Weather.HAIL in battle.weather)):
            return pick("auroraveil")
        revealed = [mv for f in foes for mv in f.moves.values() if mv.base_power > 0]
        phys = sum(mv.category == MoveCategory.PHYSICAL for mv in revealed)
        spec = sum(mv.category == MoveCategory.SPECIAL for mv in revealed)
        wanted = ("reflect", SideCondition.REFLECT) if phys > spec else ("lightscreen", SideCondition.LIGHT_SCREEN) if spec > phys else None
        if wanted and wanted[0] in avail and wanted[0] not in used and wanted[1] not in battle.side_conditions:
            return pick(wanted[0])
        return None


def make_opponent(name: str, battle_format: str) -> Player:
    """A scripted opponent that never connects to the server (the env plays its moves)."""
    cls = {"random": RandomPlayer, "maxbp": MaxBasePowerPlayer, "damage": DamageHeuristicPlayer, "tempo": TempoPlayer}.get(name)
    if cls is None:
        raise ValueError(f"unknown opponent {name!r}; choose from {OPPONENTS}")
    return cls(battle_format=battle_format, start_listening=False)
