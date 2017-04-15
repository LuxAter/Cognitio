#include <iostream>
#include <string>
#include "src/cognosco_headers.hpp"

using namespace cognosco;

// std::string goal = "Arden Rasmussen";

std::string goal =
    "It is not the strongest or the most intelligent who will survive but "
    "those who can best manage change.";

void Handle(std::pair<int, std::string> entry) {
  if (entry.first == pessum::ERROR) {
    system("setterm -fore red");
  } else if (entry.first == pessum::WARNING) {
    system("setterm -fore yellow");
  } else if (entry.first == pessum::TRACE) {
    system("setterm -fore cyan");
  }
  std::cout << entry.second << "\n";
  system("setterm -fore white");
}

std::string PrintVec(std::vector<double> vec) {
  std::string str = "<";
  for (int i = 0; i < vec.size(); i++) {
    std::stringstream ss;
    ss << vec[i];
    std::string val;
    ss >> val;
    str += val;
    if (i != vec.size() - 1) {
      str += ",";
    }
  }
  str += ">";
  return (str);
}

std::string Gen() {
  int length = rand() % 20;
  std::string str = "";
  for (int i = 0; i < length; i++) {
    str += char((rand() % 94) + 32);
  }
  return (str);
}

double Fitness(std::string val) {
  double dist = 0;
  for (int i = 0; i < val.size() && i < goal.size(); i++) {
    if (goal[i] - val[i] != 0) {
      dist += (1.0 / abs(goal[i] - val[i]));
    } else {
      dist += 10;
    }
  }
  dist -= (double)abs((int)goal.size() - (int)val.size());
  return (dist);
}

std::string CrossOver(std::string a, std::string b, double rate) {
  int pos = rand() % (int)fmin(a.size(), b.size());
  std::string end(a.begin(), a.begin() + pos);
  b.erase(b.begin(), b.begin() + pos);
  end += b;
  return (a);
}

std::string Mutate(std::string a, double rate) {
  int pos = rand() % a.size();
  if (rand() % 2 == 0) {
    a += char((rand() % 94) + 32);
  }
  if (rand() % 2 == 0) {
    a.pop_back();
  }
  a[pos] = char((rand() % 94) + 32);
  return (a);
}

int main() {
  pessum::SetLogHandle(Handle);
  srand(time(NULL));
  // Matrix<int> mata(3, 1, std::vector<int>{1, 2, 3});
  // Matrix<int> matb(1, 2, std::vector<int>{4, 5});
  // std::cout << mata.GetString() << "*" << matb.GetString() << "\n";
  // mata = Dot(mata, matb);
  // std::cout << "=" << mata.GetString();
  Network net(5, 3, 5, 5, 5, 2);
  std::cout << net.GetString() << "\n";
  std::vector<double> in = {2, 5, 7};
  std::cout << PrintVec(in) << "->";
  std::cout << PrintVec(net.ForwardProp(in)) << "\n";
  net.BackwardProp(in, {4, 1});
  pessum::SaveLog("out.log");
  return (0);
}
