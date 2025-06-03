import os
import sys
from document_cleaner import DocumentCleaner
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree

def count_files(directory: str) -> tuple:
    """Count the number of PDF files and total files in the directory."""
    pdf_count = 0
    total_count = 0
    pdf_locations = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            total_count += 1
            if file.lower().endswith('.pdf'):
                pdf_count += 1
                # Get relative path from the resources directory
                rel_path = os.path.relpath(os.path.join(root, file), directory)
                pdf_locations.append(rel_path)
                
    return pdf_count, total_count, pdf_locations

def display_resource_tree(console: Console, directory: str, pdf_files: list):
    """Display the resource directory structure with PDF files."""
    tree = Tree(f"[bold yellow]📁 {os.path.basename(directory)}")
    
    # Create a dictionary to organize files by directory
    dir_structure = {}
    for pdf_file in pdf_files:
        parts = pdf_file.split(os.sep)
        current_dict = dir_structure
        for part in parts[:-1]:  # Process directories
            if part not in current_dict:
                current_dict[part] = {}
            current_dict = current_dict[part]
        # Add file to the deepest directory
        if '_files' not in current_dict:
            current_dict['_files'] = []
        current_dict['_files'].append(parts[-1])
    
    def add_to_tree(tree, structure, indent=""):
        for name, content in sorted(structure.items()):
            if name == '_files':
                for file in sorted(content):
                    tree.add(f"[blue]📄 {file}")
            else:
                subtree = tree.add(f"[yellow]📁 {name}")
                add_to_tree(subtree, content, indent + "  ")
    
    add_to_tree(tree, dir_structure)
    console.print(tree)

def main():
    console = Console()
    
    console.print("\n[bold cyan]📚 Resource Cleaner Utility[/bold cyan]")
    console.print("This utility will process your educational resources and organize them by topic.\n")
    
    # Initialize the document cleaner
    cleaner = DocumentCleaner()
    
    # Show paths being used
    console.print("[yellow]Paths being used:[/yellow]")
    console.print(f"  • Source directory: {cleaner.source_dir}")
    console.print(f"  • Output directory: {cleaner.output_dir}\n")
    
    # Verify source directory exists
    if not os.path.exists(cleaner.source_dir):
        console.print("[red]Error: Source directory not found![/red]")
        console.print("Please make sure you have a 'resources' directory in your project root with your PDF files.")
        console.print(f"Expected location: {cleaner.source_dir}")
        return
    
    # Count files before processing
    pdf_count, total_count, pdf_locations = count_files(cleaner.source_dir)
    
    console.print(f"[yellow]Found:[/yellow]")
    console.print(f"  • {pdf_count} PDF files to process")
    console.print(f"  • {total_count} total files in resources\n")
    
    if pdf_count > 0:
        console.print("[green]Found PDF files in the following structure:[/green]")
        display_resource_tree(console, cleaner.source_dir, pdf_locations)
        console.print()
    
    if not pdf_count:
        console.print("[red]No PDF files found to process![/red]")
        console.print("Please make sure you have PDF files in your resources directory structure.")
        return
    
    # Confirm with user
    console.print("[yellow]This will create a new 'cleaned_resources' directory with organized content.[/yellow]")
    console.print("[yellow]Existing 'cleaned_resources' directory will be removed if it exists.[/yellow]\n")
    
    try:
        input("Press Enter to continue or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        console.print("\n[red]Operation cancelled by user.[/red]")
        return
    
    console.print("\n[green]Starting resource processing...[/green]")
    
    try:
        # Process the directory
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Processing files...", total=pdf_count)
            cleaner.process_directory(progress_callback=lambda: progress.advance(task))
        
        # Count processed files
        _, processed_total, processed_files = count_files(cleaner.output_dir)
        
        console.print("\n[bold green]✅ Processing complete![/bold green]")
        console.print(f"Created {processed_total} organized resource files in:")
        console.print(f"  {cleaner.output_dir}")
        
        # Print directory structure
        console.print("\n[cyan]Directory structure created:[/cyan]")
        for year in ["Year 1", "Year 2", "Year 3", "Year 4"]:
            console.print(f"\n📁 {year}")
            year_path = os.path.join(cleaner.output_dir, year)
            if os.path.exists(year_path):
                for category in sorted(os.listdir(year_path)):
                    category_path = os.path.join(year_path, category)
                    if os.path.isdir(category_path):
                        file_count = len([f for f in os.listdir(category_path) if f.endswith('.json')])
                        if file_count > 0:  # Only show non-empty directories
                            console.print(f"  └─📁 {category} ({file_count} items)")
        
    except Exception as e:
        console.print(f"\n[red]Error during processing: {str(e)}[/red]")
        return

if __name__ == "__main__":
    main() 