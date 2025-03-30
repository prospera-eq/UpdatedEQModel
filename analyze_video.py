import asyncio
from agents.video_agent import VideoAnalysisAgent
from rich.console import Console

async def main():
    console = Console()
    video_path = "video1029046881.mp4"
    
    console.print("[bold green]Initializing Video Analysis Agent...[/bold green]")
    video_agent = VideoAnalysisAgent()
    
    console.print(f"[bold blue]Starting analysis of {video_path}...[/bold blue]")
    try:
        result = await video_agent.analyze_video(video_path)
        
        console.print("\n[bold green]Analysis Results:[/bold green]")
        console.print("[cyan]Overall Emotions:[/cyan]")
        for emotion, score in result["overall_emotions"].items():
            console.print(f"  {emotion}: {score:.2f}")
            
        console.print(f"\n[cyan]Average Attention Score:[/cyan] {result['average_attention_score']:.2f}")
        console.print(f"[cyan]Frames Analyzed:[/cyan] {result['frames_analyzed']}")
        console.print(f"[cyan]Video Length:[/cyan] {result['video_length_seconds']:.1f} seconds")
        
        console.print("\n[cyan]Recommendations:[/cyan]")
        for rec in result["recommendations"]:
            console.print(f"  • {rec}")
            
    except Exception as e:
        console.print(f"[bold red]Error during analysis:[/bold red] {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
