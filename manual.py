from model.state import State
from model.action import Action

ACTION_MAP = {
    'cw': Action.ROTATE_CW,
    'ccw': Action.ROTATE_CCW,
    'l': Action.LEFT,
    'r': Action.RIGHT,
    'dh': Action.HARD_DROP,
    'ds': Action.SOFT_DROP
}


if __name__ == '__main__':
    s = State()
    while True:
        print(s)
        print(f'{s.score=}\n')
        print('\n'.join([f'{shortcut}: {ACTION_MAP[shortcut]}' for shortcut in sorted(ACTION_MAP.keys())]))
        player_action = input('>>> ')
        if player_action not in ACTION_MAP:
            print('Please select from: ' + ' '.join(sorted(ACTION_MAP.keys())))
            continue
        player_action = ACTION_MAP[player_action]
        s.step(player_action)
