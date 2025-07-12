import os
import requests
from typing import List, Dict
from openai import OpenAI
from judgeval.tracer import Tracer, wrap
from judgeval import JudgmentClient
from judgeval.data import Example
from judgeval.scorers import FaithfulnessScorer, AnswerRelevancyScorer

# Set up environment variables (add your keys here)
# os.environ["JUDGMENT_API_KEY"] = "your-judgment-api-key"
# os.environ["JUDGMENT_ORG_ID"] = "your-org-id"
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Or load from environment variables set in terminal
# Keys should be set via: export JUDGMENT_API_KEY="..." etc.

# Initialize Judgeval components
client = wrap(OpenAI())
judgment = Tracer(project_name="multi_agent_research_v2")  # Try a new project name
eval_client = JudgmentClient()

class MultiAgentResearchSystem:
    def __init__(self):
        self.research_database = []
        
    @judgment.observe(span_type="tool")
    def web_search(self, query: str) -> Dict:
        """Simulate web search - replace with real Tavily API call"""
        # For demo purposes, return mock data
        # In real implementation, use Tavily API
        mock_results = {
            "query": query,
            "results": [
                f"Research finding about {query} from source 1",
                f"Additional information on {query} from source 2",
                f"Expert analysis of {query} from source 3"
            ],
            "sources": ["source1.com", "source2.org", "source3.edu"]
        }
        return mock_results
    
    @judgment.observe(span_type="tool")
    def store_research(self, data: Dict) -> bool:
        """Store research findings in database"""
        self.research_database.append(data)
        print(f"📝 Stored research: {data['topic']}")
        return True
    
    @judgment.observe(span_type="function")
    def research_agent(self, topic: str, agent_id: int) -> Dict:
        """Individual research agent that focuses on specific topic"""
        print(f"🔍 Agent {agent_id} researching: {topic}")
        
        try:
            # Conduct web search
            search_results = self.web_search(f"{topic} research analysis")
            
            # Generate research summary using LLM
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a research agent specializing in {topic}. Provide detailed analysis based on the search results."},
                    {"role": "user", "content": f"Analyze this research data: {search_results['results']}"}
                ],
                max_tokens=300
            )
            
            if response and response.choices:
                findings = response.choices[0].message.content
            else:
                findings = f"Analysis of {topic} based on available research data."
            
            research_data = {
                "agent_id": agent_id,
                "topic": topic,
                "findings": findings,
                "sources": search_results["sources"],
                "search_query": search_results["query"]
            }
            
            # Store findings
            self.store_research(research_data)
            
            return research_data
            
        except Exception as e:
            print(f"⚠️ Research agent {agent_id} failed: {e}")
            return {
                "agent_id": agent_id,
                "topic": topic,
                "findings": f"Research failed: {str(e)}",
                "sources": [],
                "search_query": topic
            }
    
    def lead_agent(self, research_question: str) -> Dict:
        """Lead agent that coordinates research and synthesizes findings"""
        print(f"🎯 Lead Agent starting research on: {research_question}")
        
        try:
            # Break down research question into subtopics
            planning_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a research coordinator. Break down complex research questions into 3 specific subtopics for specialized agents."},
                    {"role": "user", "content": f"Break down this research question into 3 subtopics: {research_question}"}
                ],
                max_tokens=200
            )
            
            if planning_response and planning_response.choices:
                subtopics_text = planning_response.choices[0].message.content
                # Simple parsing - in real implementation, use better parsing
                subtopics = [topic.strip() for topic in subtopics_text.split('\n') if topic.strip() and not topic.strip().startswith('#')][:3]
            else:
                subtopics = []
            
            # Fallback if parsing fails
            if not subtopics:
                subtopics = ["Technical challenges", "Economic factors", "Policy considerations"]
            
            print(f"📋 Research plan: {subtopics}")
            
            # Delegate to research agents (simulate parallel execution)
            research_results = []
            for i, subtopic in enumerate(subtopics):
                try:
                    result = self.research_agent(subtopic, agent_id=i+1)
                    research_results.append(result)
                except Exception as e:
                    print(f"⚠️ Agent {i+1} failed: {e}")
                    # Continue with other agents
                    pass
            
            # Synthesize final report
            if research_results:
                all_findings = "\n".join([r["findings"] for r in research_results])
                
                synthesis_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a senior researcher. Synthesize multiple research findings into a comprehensive report."},
                        {"role": "user", "content": f"Original question: {research_question}\n\nResearch findings:\n{all_findings}\n\nCreate a comprehensive synthesis."}
                    ],
                    max_tokens=500
                )
                
                if synthesis_response and synthesis_response.choices:
                    final_synthesis = synthesis_response.choices[0].message.content
                else:
                    final_synthesis = f"Synthesis of research on: {research_question}\n\nBased on findings from {len(research_results)} research agents."
            else:
                final_synthesis = "Research could not be completed due to technical issues."
            
            final_report = {
                "research_question": research_question,
                "subtopics": subtopics,
                "individual_research": research_results,
                "final_synthesis": final_synthesis,
                "total_agents_used": len(research_results)
            }
            
            return final_report
            
        except Exception as e:
            print(f"⚠️ Lead agent failed: {e}")
            # Return minimal report instead of None
            return {
                "research_question": research_question,
                "subtopics": ["Error occurred"],
                "individual_research": [],
                "final_synthesis": f"Research failed due to error: {str(e)}",
                "total_agents_used": 0
            }
    
    def evaluate_research_quality(self, report: Dict) -> Dict:
        """Evaluate the quality of research using Judgeval"""
        print("🧪 Evaluating research quality...")
        
        # Create evaluation example
        example = Example(
            input=report["research_question"],
            actual_output=report["final_synthesis"],
            retrieval_context=[r["findings"] for r in report["individual_research"]]
        )
        
        # Set up scorers
        scorers = [
            FaithfulnessScorer(threshold=0.7),
            AnswerRelevancyScorer(threshold=0.6)
        ]
        
        # Run evaluation
        try:
            results = eval_client.run_evaluation(
                examples=[example],
                scorers=scorers,
                model="gpt-3.5-turbo",
                project_name="multi_agent_research_eval"
            )
            
            evaluation_summary = {
                "evaluation_success": results[0].success,
                "scores": {scorer.name: scorer.score for scorer in results[0].scorers_data},
                "evaluation_details": results[0].scorers_data
            }
            
            return evaluation_summary
            
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {"evaluation_success": False, "error": str(e)}

def main():
    """Main function to run the research system"""
    system = MultiAgentResearchSystem()
    
    # Example research question
    research_question = "What are the main challenges and opportunities for renewable energy adoption in developing countries?"
    
    print("🚀 Starting Multi-Agent Research System")
    print("=" * 50)
    
    # Run the research
    report = system.lead_agent(research_question)
    
    print("\n📊 Research Complete!")
    print("=" * 50)
    print(f"Question: {report['research_question']}")
    print(f"Agents Used: {report['total_agents_used']}")
    print(f"Subtopics: {report['subtopics']}")
    print("\n📝 Final Synthesis:")
    print(report['final_synthesis'])
    
    # Evaluate the research
    evaluation = system.evaluate_research_quality(report)
    
    print("\n🧪 Evaluation Results:")
    print("=" * 50)
    if evaluation["evaluation_success"]:
        for scorer_name, score in evaluation["scores"].items():
            print(f"{scorer_name}: {score}")
    else:
        print(f"Evaluation failed: {evaluation.get('error', 'Unknown error')}")
    
    return report, evaluation

if __name__ == "__main__":
    report, evaluation = main()