def add_arguments(parser):
    parser.add_argument('--env-mode', default='full', help='env mode to use: full | part | skip (full)')
    parser.add_argument('--env-envparas', default='', help='myenv parameters')
    parser.add_argument('--env-npcparas', default='', help='npc units parameters')
    parser.add_argument('--env-agtparas', default='', help='agt units parameters')
    parser.add_argument('--env-envonoff', default='', help='env switches')
    parser.add_argument('--env-pobparas', default='', help='pomdp parameters')
def add_strings(args):
    args.exp_dir=args.exp_dir+':'+args.env_mode+'_'+args.env_envparas+'_'+args.env_npcparas+'_'+args.env_agtparas+'_'+args.env_envonoff \
                                               +'_'+args.env_pobparas
import importlib,myenv
def getEnvir(args):
    mod = importlib.import_module('envirs.'+args.env_mode)
    return mod.fEnv(args)
