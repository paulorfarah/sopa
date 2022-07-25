from pydriller import Repository
from pydriller.git import Git

commits = Git.get_commit_from_tag("release")
repo_path = "https://github.com/paulorfarah/maven_project"
for commit in Repository(repo_path, to_tag='').traverse_commits():
    print(commit)