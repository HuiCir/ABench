"""ATIF contract regressions: sequential steps and tool-result links follow Harbor's reference schema."""
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from agents.trajectory import claude_to_atif


class AtifTest(unittest.TestCase):
    def test_messages_calls_observations_and_metrics(self):
        events = [
            {'type':'assistant','session_id':'session-1','message':{'id':'m1','model':'model-1','content':[
                {'type':'text','text':'Read the file.'},
                {'type':'tool_use','id':'call-1','name':'Read','input':{'file_path':'a.py'}}],
                'usage':{'input_tokens':3,'cache_read_input_tokens':4,'cache_creation_input_tokens':2,'output_tokens':5}}},
            {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'call-1','content':'file contents'}]}},
            {'type':'assistant','message':{'id':'m2','model':'model-1','content':[{'type':'text','text':'Done.'}]}},
            {'type':'result','total_cost_usd':0.01,'usage':{'input_tokens':6,'output_tokens':7}}]
        raw = b'\n'.join(json.dumps(e).encode() for e in events)
        output = claude_to_atif(raw,'2.1.38',prompt='Inspect a.py')
        self.assertEqual(output['schema_version'],'ATIF-v1.8')
        self.assertEqual(output['agent'],{'name':'claude-code','version':'2.1.38','model_name':'model-1'})
        self.assertEqual([s['step_id'] for s in output['steps']],[1,2,3])
        self.assertEqual([s['source'] for s in output['steps']],['user','agent','agent'])
        call_step = output['steps'][1]
        self.assertEqual(call_step['tool_calls'][0]['arguments'],{'file_path':'a.py'})
        self.assertEqual(call_step['observation']['results'][0]['source_call_id'],call_step['tool_calls'][0]['tool_call_id'])
        self.assertEqual(call_step['metrics']['prompt_tokens'],9)
        self.assertEqual(output['final_metrics']['total_completion_tokens'],7)
        self.assertEqual(output['final_metrics']['total_cost_usd'],0.01)
        self.assertNotIn('timestamp',call_step)

    def test_repeated_message_id_and_orphan_observation(self):
        event={'type':'assistant','message':{'id':'one','content':[{'type':'text','text':'one response'}]}}
        orphan={'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'missing','content':'unknown'}]}}
        out=claude_to_atif('\n'.join(json.dumps(e) for e in (event,event,orphan)).encode(),'2.1.38')
        self.assertEqual(len(out['steps']),2)
        self.assertEqual(out['steps'][0]['message'],'one response')
        self.assertNotIn('source_call_id',out['steps'][1]['observation']['results'][0])
        self.assertEqual(out['steps'][1]['observation']['results'][0]['extra']['original_call_id'],'missing')
        self.assertEqual(out['session_id'],claude_to_atif('\n'.join(json.dumps(e) for e in (event,event,orphan)).encode(),'2.1.38')['session_id'])
